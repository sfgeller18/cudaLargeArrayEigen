#ifndef SHIFT_HPP
#define SHIFT_HPP

#include <vector>
#include <complex>
#include <cstddef>

// #include "structs.hpp"
#include "utils.hpp"

#include "eigenSolver.hpp"
#include "arnoldi.hpp"
#include "cuda_manager.hpp"

// #define CUBLAS_RESTART
#define EIGEN_RESTART

#ifdef CUBLAS_RESTART
inline int cublasComputeQ(DeviceComplexType* d_Q, std::vector<DeviceComplexType*> h_Tauarray, ComplexKrylovPair& q_h, const ComplexVector& eigenvalues, const size_t& basis_size, cublasHandle_t& handle, cusolverDnHandle_t& solver_handle) {
    const size_t N = q_h.m;
    const size_t N_squared = N * N;
    const size_t batch_count = N - basis_size;

    int info;
    int* d_info = cudaMallocChecked<int>(sizeof(int));
    
    const ComplexMatrix& H = q_h.H;
    const ComplexMatrix& V = q_h.Q;
    size_t V_Rows = V.rows();



    // Allocate device memory
    DeviceComplexType* d_matrices = cudaMallocChecked<DeviceComplexType>((batch_count + 1) * N_squared * sizeof(DeviceComplexType));
    DeviceComplexType* d_tau = cudaMallocChecked<DeviceComplexType>(batch_count * N * sizeof(DeviceComplexType));
    DeviceComplexType** d_Tauarray = cudaMallocChecked<DeviceComplexType*>(batch_count * sizeof(DeviceComplexType*));

    DeviceComplexType** d_Aarray = cudaMallocChecked<DeviceComplexType*>(batch_count * sizeof(DeviceComplexType*));

    
    // Host arrays for device pointers
    std::vector<DeviceComplexType*> h_Aarray(batch_count);

    // Sort eigenvalues by magnitude
    std::vector<size_t> indices(batch_count);
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(),
            [&eigenvalues](size_t i1, size_t i2) { 
                return std::norm(eigenvalues(i1)) < std::norm(eigenvalues(i2)); 
            });

    // Prepare matrices on host (+ true copy of d_H to modify)
    std::vector<DeviceComplexType> h_matrices((batch_count + 1) * N_squared);
    for (int k = 0; k < batch_count + 1; k++) {
        for (int j = 0; j < N; j++) {
            for (int i = 0; i < N; i++) {
                size_t idx = k * N_squared + j * N + i;
                if ((i != j) or (k == batch_count)) {
                    h_matrices[idx] = toDeviceComplex(H(i,j));
                } else {
                    h_matrices[idx] = toDeviceComplex(H(j,i) - eigenvalues(indices[k]));
                }
            }
        }
    }

    // Copy matrices to device
    cudaMemcpyChecked(d_matrices, h_matrices.data(), (batch_count + 1) * N_squared * sizeof(DeviceComplexType), cudaMemcpyHostToDevice);

    // Set up device pointer arrays
    for (int i = 0; i < batch_count; i++) {
        h_Aarray[i] = d_matrices + i * N_squared;
        h_Tauarray[i] = d_tau + i * N;
    }

    // Copy pointer arrays to device
    cudaMemcpyChecked(d_Aarray, h_Aarray.data(), batch_count * sizeof(DeviceComplexType*), cudaMemcpyHostToDevice);
    cudaMemcpyChecked(d_Tauarray, h_Tauarray.data(), batch_count * sizeof(DeviceComplexType*), cudaMemcpyHostToDevice);

    // Perform batch QR factorization
    CHECK_CUBLAS(cublasZgeqrfBatched(handle, N, N, d_Aarray, N, d_Tauarray, &info, batch_count));

    CHECK_CUDA(cudaMemcpy2D(
        d_Q,                          // dst
        N * sizeof(DeviceComplexType), // dst pitch (width of each row in bytes)
        d_matrices,                    // src
        N * sizeof(DeviceComplexType), // src pitch
        N * sizeof(DeviceComplexType), // width in bytes to copy per row
        N * batch_count,               // number of rows
        cudaMemcpyDeviceToDevice));

    
    // Cleanup device memory
    cudaFreeChecked(d_matrices);
    cudaFreeChecked(d_tau);
    cudaFreeChecked(d_Aarray);
    cudaFreeChecked(d_Tauarray);
    cudaFreeChecked(d_info);

    return 0;
}

int cublasQProduct(DeviceComplexType* d_Q, std::vector<DeviceComplexType*> h_Tauarray, ComplexKrylovPair& q_h, const size_t& basis_size, cublasHandle_t& handle, cusolverDnHandle_t& solver_handle) {
    const size_t N = q_h.m;
    const size_t N_squared = N * N;
    const size_t batch_count = N - basis_size;
    const ComplexMatrix& V = q_h.Q;
    size_t V_Rows = V.rows();

    constexpr DeviceComplexType one = cuDoubleComplex(1.0, 0.0);
    constexpr DeviceComplexType zero = cuDoubleComplex(0.0, 0.0);

    int info;
    int* d_info = cudaMallocChecked<int>(sizeof(int));

    int lwork;
    ComplexMatrix V_copy = q_h.Q;

    DeviceComplexType* d_H = cudaMallocChecked<DeviceComplexType>(N * N * sizeof(DeviceComplexType));
    DeviceComplexType* d_V = cudaMallocChecked<DeviceComplexType>(V_Rows * N * sizeof(DeviceComplexType));

    cudaMemcpyChecked(d_V, V_copy.data(), V_Rows * N * sizeof(DeviceComplexType), cudaMemcpyHostToDevice);


    CHECK_CUSOLVER(cusolverDnZungqr_bufferSize(solver_handle, N, N, N, d_Q, N, h_Tauarray[0], &lwork));
    DeviceComplexType* d_work = cudaMallocChecked<DeviceComplexType>(lwork * sizeof(DeviceComplexType));

    // Copy all QR factorization matrices to device in one go

    // Process each QR factorization. Use Zgemm to compute d_H = d_S + d_Q^H * d_H * d_Q, d_V = d_V * d_Q
    for (int batch_idx = 0; batch_idx < batch_count; batch_idx++) {
        // Each d_Q[batch_idx] is a pointer to the batch_idx-th matrix
        DeviceComplexType* d_Q_current = d_Q + batch_idx * N * N; // Pointer to the current matrix
        CHECK_CUSOLVER(cusolverDnZungqr(solver_handle, N, N, N, d_Q_current, N, h_Tauarray[batch_idx], d_work, lwork, d_info));
        CHECK_CUBLAS(cublasZgemm(handle, CUBLAS_OP_C, CUBLAS_OP_N, N, N, N, &one, d_Q_current, N, d_H, N, &zero, d_H, N));
        CHECK_CUBLAS(cublasZgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &one, d_H, N, d_Q_current, N, &zero, d_H, N));
        CHECK_CUBLAS(cublasZgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, V_Rows, N, N, &one, d_V, V_Rows, d_Q_current, N, &zero, d_V, V_Rows));
    }


    // #define DEBUG
    #ifdef DEBUG
    ComplexMatrix S = ComplexMatrix::Identity(N,N);
    ComplexMatrix h_Q(N, N * batch_count);    
    ComplexMatrix h_h = q_h.H;
    std::cout << h_h << std::endl;
    std::vector<DeviceComplexType> buffer(N * N * batch_count);
    cudaMemcpyChecked(buffer.data(), d_Q, N * N * batch_count * sizeof(DeviceComplexType), cudaMemcpyDeviceToHost);
    h_Q = Eigen::Map<ComplexMatrix>(reinterpret_cast<std::complex<double>*>(buffer.data()), N, N * batch_count);
    for (int i = 0; i < batch_count; i++) {
        // Extract the current batch
        ComplexMatrix h_Q_current = h_Q.block(0, i * N, N, N);
        S *= h_Q_current;
        mollify(h_Q_current);
        h_h = h_Q_current.adjoint() * h_h * h_Q_current;
        mollify(h_h);
        std::cout << h_h << std::endl;
        // assert(isHessenberg<ComplexMatrix>(h_h));
        assert(isHessenberg<ComplexMatrix>(h_Q_current));
        // assert(isHessenberg<ComplexMatrix>(S));
    }
    #endif



    // Copy result back to host
    cudaMemcpyChecked(q_h.H.data(), d_H, N * N * sizeof(DeviceComplexType), cudaMemcpyDeviceToHost);
    cudaMemcpyChecked(q_h.Q.data(), d_V, V_Rows * N * sizeof(DeviceComplexType), cudaMemcpyDeviceToHost);

        // Cleanup device memory
    cudaFreeChecked(d_V);
    cudaFreeChecked(d_H);
    cudaFreeChecked(d_info);
    cudaFreeChecked(d_work);

    return 0;
    }

inline int cublasQRShift(ComplexKrylovPair& q_h, const ComplexVector& eigenvalues, const size_t& basis_size, cublasHandle_t& handle, cusolverDnHandle_t& solver_handle) {
    const size_t N = q_h.m;
    const size_t N_squared = N * N;
    const size_t batch_count = N - basis_size;

    // Allocs for cross-function arrays
    DeviceComplexType* d_Q = cudaMallocChecked<DeviceComplexType>((batch_count + 1) * N_squared * sizeof(DeviceComplexType));
    std::vector<DeviceComplexType*> h_Tauarray(batch_count);
    DeviceComplexType* d_tau = cudaMallocChecked<DeviceComplexType>(batch_count * N * sizeof(DeviceComplexType));    
    for (size_t i = 0; i < batch_count; ++i) {h_Tauarray[i] = d_tau + i * N;}

    cublasComputeQ(d_Q, h_Tauarray, q_h, eigenvalues, basis_size, handle, solver_handle);
    cublasQProduct(d_Q, h_Tauarray, q_h, basis_size, handle, solver_handle);

    cudaFreeChecked(d_Q);
    cudaFreeChecked(d_tau);
    return 0;
}

#endif //GPU_RESTART

// Pair must be passed as Complex Matrix. Modified in Place
int reduceArnoldiPair(ComplexKrylovPair& q_h, const size_t& basis_size, cublasHandle_t& handle, cusolverDnHandle_t& solver_handle, const resize_type resize_method) {
    // Compute eigenvalues and eigenvectors
    assert(q_h.m > basis_size);
    ComplexMatrix S = ComplexMatrix::Identity(q_h.m, q_h.m);
    ComplexMatrix& H = q_h.H;
    ComplexMatrix& Q = q_h.Q;
    EigenPairs H_pairs{};

    // Start timing for eigsolver
    auto start = std::chrono::high_resolution_clock::now();
    hessEigSolver<ComplexMatrix>(H, H_pairs, q_h.m);
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Time for eigsolver: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
              << " ms" << std::endl;

    double tol = 1e-10 * H.norm();

    // Start timing for incremental shifts
    start = std::chrono::high_resolution_clock::now();
    #ifdef EIGEN_RESTART
    Eigen::MatrixXcd Qi(q_h.m, q_h.m);
    for (int i = 0; i < q_h.m - basis_size; i++) {
        Eigen::HouseholderQR<Eigen::MatrixXcd> qr(H - H_pairs.values(i) * Eigen::MatrixXcd::Identity(q_h.m, q_h.m));
        Qi = qr.householderQ();
        H = Qi.adjoint() * H * Qi;
        S *= Qi;
        mollify(H, tol);
        mollify(Q, tol);
    }
    #endif
    #ifdef CUBLAS_RESTART
    cublasQRShift(q_h, H_pairs.values, basis_size, handle, solver_handle);
    #endif
    mollify(q_h.H);
    // std::cout << q_h.H <<std::endl;
    end = std::chrono::high_resolution_clock::now();
    std::cout << "Time for incremental shifts: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
              << " ms" << std::endl;

    assert(isHessenberg<ComplexMatrix>(H));

    // Start timing for resizing
    start = std::chrono::high_resolution_clock::now();
    if (resize_method == resize_type::ZEROS) {
        H.bottomRows(q_h.m - basis_size).setZero();
        H.rightCols(q_h.m - basis_size).setZero();
        Q.rightCols(q_h.m - basis_size).setZero();
    } else {
        H.conservativeResize(basis_size, basis_size);
        Q.conservativeResize(q_h.m, basis_size);
        q_h.m = basis_size;
    }
    end = std::chrono::high_resolution_clock::now();
    std::cout << "Time for resizing: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
              << " ms" << std::endl;

    mollify(Q);
    mollify(H);

    return 0;
}

int constructSMatrix(cusolverDnHandle_t solver_handle,
                            cublasHandle_t blas_handle,
                            const std::vector<DeviceComplexType*>& h_Aarray, 
                            const std::vector<DeviceComplexType*>& h_Tauarray,
                            const int m, // Basis Size
                            const int n, // Q column length (Original Space Size)
                            const int lda,
                            ComplexMatrix& S,
                            const int batch_count) {

    // Allocate device memory for Q matrix and result
    const DeviceComplexType one = make_cuDoubleComplex(1.0, 0.0);
    const DeviceComplexType zero = make_cuDoubleComplex(0.0, 0.0);
    DeviceComplexType* d_Q = cudaMallocChecked<DeviceComplexType>(m * m * sizeof(DeviceComplexType));
    DeviceComplexType* d_S = cudaMallocChecked<DeviceComplexType>(m * m * sizeof(DeviceComplexType));
    DeviceComplexType* d_temp = cudaMallocChecked<DeviceComplexType>(m * m * sizeof(DeviceComplexType));
    int* d_info = cudaMallocChecked<int>(sizeof(int));
    
    int lwork;
    CHECK_CUSOLVER(cusolverDnZungqr_bufferSize(solver_handle, m, m, std::min(m,n), d_Q, lda, h_Tauarray[0], &lwork));
    DeviceComplexType* d_work = cudaMallocChecked<DeviceComplexType>(lwork * sizeof(DeviceComplexType));


    ComplexMatrix Q(m, n);
    for (int batch_idx = 0; batch_idx < batch_count; batch_idx++) {
        cudaMemcpyChecked(d_Q, h_Aarray[batch_idx], m * m * sizeof(DeviceComplexType), cudaMemcpyDeviceToDevice);        
        CHECK_CUSOLVER(cusolverDnZungqr(solver_handle, m, m, std::min(m,n), d_Q, lda, h_Tauarray[batch_idx], d_work, lwork, d_info));
        if (batch_idx == 0) {cudaMemcpyChecked(d_S, d_Q, m * m * sizeof(DeviceComplexType), cudaMemcpyDeviceToDevice);}
        else {CHECK_CUBLAS(cublasZgemm(blas_handle, CUBLAS_OP_N, CUBLAS_OP_N, m, m, m, &one, d_S, m, d_Q, m, &zero, d_S, m));}
    }

    cudaMemcpyChecked(S.data(), d_S, m * m * sizeof(DeviceComplexType), cudaMemcpyDeviceToHost);

    assert(isOrthonormal<ComplexMatrix>(S));
    
    cudaFreeChecked(d_Q);
    cudaFreeChecked(d_S);
    cudaFreeChecked(d_temp);
    cudaFreeChecked(d_info);
    cudaFreeChecked(d_work);

    return 0;
}
#endif // SHIFT_HPP