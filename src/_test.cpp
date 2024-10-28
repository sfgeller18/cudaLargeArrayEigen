#include <cassert>
#include <cmath>
#include <string>
#include "tests.hpp"
#include "cuda_manager.hpp"

// #define DEVICE_SHIFT

using MatrixType = MatrixColMajor;

inline int constructSMatrix(cusolverDnHandle_t solver_handle,
                          const std::vector<DeviceComplexType*>& h_Aarray, 
                          const std::vector<DeviceComplexType*>& h_Tauarray,
                          const int m,
                          const int n,
                          const int lda,
                          ComplexMatrix& S,
                          const int batch_count) {
    // Allocate device memory for Q matrix
    DeviceComplexType* d_Q;
    ComplexMatrix Q(m, m);
    int* d_info;

    //I believe this is fine as all the arrays keep the same size
    DeviceComplexType* d_work;
    int lwork;
    CHECK_CUSOLVER(cusolverDnZungqr_bufferSize(
        solver_handle, m, m, std::min(m,n),
        d_Q, lda,
        h_Tauarray[0],
        &lwork));
    CHECK_CUDA(cudaMalloc(&d_work, lwork * sizeof(DeviceComplexType)));

    CHECK_CUDA(cudaMalloc(&d_Q, m * n * sizeof(DeviceComplexType)));
    CHECK_CUDA(cudaMalloc(&d_info, sizeof(int)));

    for (int batch_idx = 0; batch_idx < batch_count; batch_idx++) {
        // Copy the QR factored matrix to d_Q
        CHECK_CUDA(cudaMemcpy(d_Q, h_Aarray[batch_idx], 
                             m * n * sizeof(DeviceComplexType),
                             cudaMemcpyDeviceToDevice));
        
        // Construct Q matrix from the QR factorization
        CHECK_CUSOLVER(cusolverDnZungqr(
            solver_handle, m, m, std::min(m,n),
            d_Q, lda,
            h_Tauarray[batch_idx],
            d_work, lwork,
            d_info));
        
        // Copy result back to host
        CHECK_CUDA(cudaMemcpy(Q.data(), d_Q,
                             m * m * sizeof(DeviceComplexType),
                             cudaMemcpyDeviceToHost));
           if (batch_idx == 0) {
                S = Q;
            } else {
                S = S * Q;
            }
        
        // Free workspace for next iteration
    }
    
    // Clean up
    CHECK_CUDA(cudaFree(d_Q));
    CHECK_CUDA(cudaFree(d_info));
    CHECK_CUDA(cudaFree(d_work));
    return 0;
}

int computeShift(ComplexMatrix& S,
                 const ComplexMatrix& complex_M, 
                 const ComplexVector& eigenvalues,
                 const size_t& N,
                 const size_t& m) {
    const size_t& N_squared = N * N;
    const size_t& batch_count = N - m;

    // Create cublas handle and perform QR factorization
    cublasHandle_t handle;
    cusolverDnHandle_t solver_handle;
    CHECK_CUBLAS(cublasCreate(&handle));
    CHECK_CUSOLVER(cusolverDnCreate(&solver_handle));

    // Allocate device memory for matrices and tau
    DeviceComplexType* d_matrices;
    DeviceComplexType* d_tau;
    CHECK_CUDA(cudaMalloc(&d_matrices, batch_count * N_squared * sizeof(DeviceComplexType)));
    CHECK_CUDA(cudaMalloc(&d_tau, batch_count * N * sizeof(DeviceComplexType)));

    // Allocate array of pointers
    DeviceComplexType** d_Aarray;
    DeviceComplexType** d_Tauarray;
    CHECK_CUDA(cudaMalloc(&d_Aarray, batch_count * sizeof(DeviceComplexType*)));
    CHECK_CUDA(cudaMalloc(&d_Tauarray, batch_count * sizeof(DeviceComplexType*)));

    std::vector<size_t> indices(batch_count);
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(),
              [&eigenvalues](size_t i1, size_t i2) { return std::norm(eigenvalues[i1]) < std::norm(eigenvalues[i2]); });

    ComplexVector smallest_eigenvalues(batch_count);
    for (size_t i = 0; i < batch_count; ++i) {
        smallest_eigenvalues[i] = eigenvalues[indices[i]];
    }

    // Prepare matrices on host
    std::vector<DeviceComplexType> h_matrices(batch_count * N_squared);
    for (int k = 0; k < batch_count; k++) {
        for (int j = 0; j < N; j++) {
            for (int i = 0; i < N; i++) {
                size_t idx = k * N_squared + j * N + i;
                if (i == j) {
                    h_matrices[idx] = toDeviceComplex(complex_M(j,i) - smallest_eigenvalues[k]);
                } else {
                    h_matrices[idx] = toDeviceComplex(complex_M(j,i));
                }
            }
        }
    }

    // Copy matrices to device
    CHECK_CUDA(cudaMemcpy(d_matrices, h_matrices.data(), 
                         batch_count * N_squared * sizeof(DeviceComplexType), 
                         cudaMemcpyHostToDevice));

    // Set up arrays of pointers
    std::vector<DeviceComplexType*> h_Aarray(batch_count);
    std::vector<DeviceComplexType*> h_Tauarray(batch_count);
    for (int i = 0; i < batch_count; i++) {
        h_Aarray[i] = d_matrices + i * N_squared;
        h_Tauarray[i] = d_tau + i * N;
    }

    // Copy pointer arrays to device
    CHECK_CUDA(cudaMemcpy(d_Aarray, h_Aarray.data(), batch_count * sizeof(DeviceComplexType*), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_Tauarray, h_Tauarray.data(), batch_count * sizeof(DeviceComplexType*), cudaMemcpyHostToDevice));

    // Perform batched QR factorization
    int info;
    CHECK_CUBLAS(cublasZgeqrfBatched(handle, N, N, d_Aarray, N, d_Tauarray, &info, batch_count));

    // Pass the device array pointer and host tau array
    constructSMatrix(solver_handle, h_Aarray, h_Tauarray, N, N, N, S, batch_count);
    
    checkOrthonormality<ComplexMatrix>(S);

    // Clean up
    CHECK_CUDA(cudaFree(d_matrices));
    CHECK_CUDA(cudaFree(d_tau));
    CHECK_CUDA(cudaFree(d_Aarray));
    CHECK_CUDA(cudaFree(d_Tauarray));

    CHECK_CUBLAS(cublasDestroy(handle));
    CHECK_CUSOLVER(cusolverDnDestroy(solver_handle));

    return 0;
}


int reduceEvals(ComplexMatrix& H, ComplexMatrix& Q, const size_t& N, const size_t& k) {
    print(Q);
    print(H);
    ComplexVector evals(N);
    ComplexMatrix evecs(N,N);
    eigsolver(H, evals, evecs);
    ComplexMatrix S(N,N);
    computeShift(S, H, evals, N, k);
    Q = Q * S;
    H = S.adjoint() * H * S; 

    print(Q);
    print(H);
    return 0;
}

int main() {
    const size_t N = 20;
    const size_t k = 5;
    ComplexVector evals(N);
    ComplexMatrix evecs(N,N);
    ComplexMatrix H = generateRandomHessenbergMatrix<ComplexMatrix>(N);
    ComplexMatrix V = gramSchmidtOrthonormal(N);

    eigsolver(H, evals, evecs);
    reduceEvals(H, V, N, k);
    print(V);
}

