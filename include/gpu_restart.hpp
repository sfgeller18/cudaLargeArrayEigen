#include <iostream>
#include <vector>
#include <algorithm>
#include <cassert>
#include "arnoldi.hpp"
#include "tests.hpp"
#include <chrono>
#include "cuda_manager.hpp"

inline int cublasQRShift(ComplexKrylovPair& q_h, const ComplexVector& eigenvalues, const size_t& basis_size, cublasHandle_t& handle, cusolverDnHandle_t& solver_handle) {
    const size_t N = q_h.m;
    const size_t N_squared = N * N;
    const size_t batch_count = N - basis_size;
    const DeviceComplexType one = make_cuDoubleComplex(1.0, 0.0);
    const DeviceComplexType zero = make_cuDoubleComplex(0.0, 0.0);
    int info;
    int* d_info = cudaMallocChecked<int>(sizeof(int));
    int lwork;

    ComplexMatrix V_copy = q_h.Q;
    const ComplexMatrix& H = q_h.H;
    const ComplexMatrix& V = q_h.Q;
    size_t V_Rows = V.rows();



    DeviceComplexType* d_H = cudaMallocChecked<DeviceComplexType>(N * N * sizeof(DeviceComplexType));
    DeviceComplexType* d_Q = cudaMallocChecked<DeviceComplexType>(batch_count * N * N * sizeof(DeviceComplexType));
    DeviceComplexType* d_V = cudaMallocChecked<DeviceComplexType>(V_Rows * N * sizeof(DeviceComplexType));
    cudaMemcpyChecked(d_V, V_copy.data(), V_Rows * N * sizeof(DeviceComplexType), cudaMemcpyHostToDevice);

    // Allocate device memory
    DeviceComplexType* d_matrices = cudaMallocChecked<DeviceComplexType>((batch_count + 1) * N_squared * sizeof(DeviceComplexType));
    DeviceComplexType* d_tau = cudaMallocChecked<DeviceComplexType>(batch_count * N * sizeof(DeviceComplexType));
    DeviceComplexType** d_Aarray = cudaMallocChecked<DeviceComplexType*>(batch_count * sizeof(DeviceComplexType*));
    DeviceComplexType** d_Tauarray = cudaMallocChecked<DeviceComplexType*>(batch_count * sizeof(DeviceComplexType*));

    
    // Host arrays for device pointers
    std::vector<DeviceComplexType*> h_Aarray(batch_count);
    std::vector<DeviceComplexType*> h_Tauarray(batch_count);

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

    

    // Get workspace size and allocate
    CHECK_CUSOLVER(cusolverDnZungqr_bufferSize(solver_handle, N, N, N, d_Q, N, h_Tauarray[0], &lwork));
    DeviceComplexType* d_work = cudaMallocChecked<DeviceComplexType>(lwork * sizeof(DeviceComplexType));

    // Copy all QR factorization matrices to device in one go
    CHECK_CUDA(cudaMemcpy2D(
        d_Q,                          // dst
        N * sizeof(DeviceComplexType), // dst pitch (width of each row in bytes)
        d_matrices,                    // src
        N * sizeof(DeviceComplexType), // src pitch
        N * sizeof(DeviceComplexType), // width in bytes to copy per row
        N * batch_count,               // number of rows
        cudaMemcpyDeviceToDevice));

    // Process each QR factorization
    for (int batch_idx = 0; batch_idx < batch_count; batch_idx++) {
        // Each d_Q[batch_idx] is a pointer to the batch_idx-th matrix
        DeviceComplexType* d_Q_current = d_Q + batch_idx * N * N; // Pointer to the current matrix
        CHECK_CUSOLVER(cusolverDnZungqr(solver_handle, N, N, N, d_Q_current, N, h_Tauarray[batch_idx], d_work, lwork, d_info));

        // Use Zgemm to compute d_H = d_S + d_Q^H * d_H * d_Q, d_V = d_V * d_Q
        CHECK_CUBLAS(cublasZgemm(handle, CUBLAS_OP_C, CUBLAS_OP_N, N, N, N, &one, d_Q_current, N, d_H, N, &zero, d_H, N));
        CHECK_CUBLAS(cublasZgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &one, d_H, N, d_Q_current, N, &zero, d_H, N));
        CHECK_CUBLAS(cublasZgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, V_Rows, N, N, &one, d_V, V_Rows, d_Q_current, N, &zero, d_V, V_Rows));

    }


    #define DEBUG
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
    cudaFreeChecked(d_Q);
    cudaFreeChecked(d_V);
    cudaFreeChecked(d_H);
    cudaFreeChecked(d_info);
    cudaFreeChecked(d_work);

    // Cleanup remaining device memory
    cudaFreeChecked(d_matrices);
    cudaFreeChecked(d_tau);
    cudaFreeChecked(d_Aarray);
    cudaFreeChecked(d_Tauarray);

    return 0;
    }