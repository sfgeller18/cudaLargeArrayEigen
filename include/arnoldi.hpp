#ifndef ARNOLDI_HPP
#define ARNOLDI_HPP

#include <array>
#include <tuple>
#include <iostream>
#include <vector>
#include "matmul.hpp"
#include "cuda_manager.hpp"
#include "vector.hpp"
#include "eigenSolver.hpp"

constexpr size_t MAX_EVEC_ON_DEVICE = 1000;

struct ArnoldiPair { MatrixColMajor Q; MatrixColMajor H_tilde; size_t m;};

template <typename MatrixType>
ArnoldiPair arnoldiEigen(const MatrixType& M, const size_t& max_iters, const HostPrecision& tol) {
    size_t N, L; //N=num_rows, L=num_cols
    std::tie(N, L) = shape(M);
    Vector v0 = randVecGen(N);
    Vector h_evals(max_iters);
    #ifdef USE_EIGEN
        h_evals[0] = v0.norm();
        v0.normalize();
    #else
        h_evals[0] = norm(v0);
        for (HostPrecision v : v0) {v /= h_evals[0];}      //v0 is a norm 1 random vector
    #endif


    size_t free_mem = 0;
    size_t total_mem = 0;
    cudaError_t error = cudaMemGetInfo(&free_mem, &total_mem);
    const size_t NUM_EVECS_ON_DEVICE = std::min(MAX_EVEC_ON_DEVICE, max_iters + 1);
    size_t ROWS = MAX_ROW_ALLOC(free_mem, N);

    size_t m = 1;


    // Allocations
// Allocations
DevicePrecision* d_evecs = cudaMallocChecked<DevicePrecision>(NUM_EVECS_ON_DEVICE * N * PRECISION_SIZE);
DevicePrecision* d_proj = cudaMallocChecked<DevicePrecision>(NUM_EVECS_ON_DEVICE * PRECISION_SIZE);
DevicePrecision* d_y = cudaMallocChecked<DevicePrecision>(N * PRECISION_SIZE);
DevicePrecision* d_M = cudaMallocChecked<DevicePrecision>(ROWS * N * PRECISION_SIZE);
DevicePrecision* d_result = cudaMallocChecked<DevicePrecision>(N * PRECISION_SIZE);  // Changed L to N
DevicePrecision* d_h = cudaMallocChecked<DevicePrecision>((max_iters + 1) * max_iters * PRECISION_SIZE);  // Corrected size

Vector norms(max_iters);

// Initial setup
cudaMemcpyChecked(d_y, v0.data(), N * PRECISION_SIZE, cudaMemcpyHostToDevice);
cudaMemcpyChecked(d_evecs, v0.data(), N * PRECISION_SIZE, cudaMemcpyHostToDevice);

cublasHandle_t handle;
cublasCreate(&handle);

constexpr DevicePrecision alpha = 1.0;
constexpr DevicePrecision beta = 0.0;
constexpr DevicePrecision neg_one = -1.0;

std::cout << "HAHA" << std::endl;

for (int i = 0; i < max_iters; i++) {  // Changed to start from 0
    matmul_internal<MatrixType>(M, d_M, d_y, d_result, ROWS, N, L, handle);
    MatrixType M_holder(N, L);
    cudaMemcpyChecked(M_holder.data(), d_M, N * L * PRECISION_SIZE, cudaMemcpyDeviceToHost);
    // Compute co-linearities w/ previous basis vectors
    cublasGemv(handle, CUBLAS_OP_T, N, i + 1, &alpha, d_evecs, N, d_result, 1, &beta, &d_h[i * (max_iters + 1)], 1);

    // Subtract co-linearities from new result vector
    cublasGemv(handle, CUBLAS_OP_N, N, i + 1, &neg_one, d_evecs, N, &d_h[i * (max_iters + 1)], 1, &alpha, d_result, 1);

    MatrixType h_res(max_iters + 1, max_iters);
    cudaMemcpyChecked(h_res.data(), d_h,  (max_iters + 1) * max_iters * sizeof(HostPrecision), cudaMemcpyDeviceToHost);
    // print(h_res);

    // TO-DO: FUCNTION PTR THIS BAD BOY
    #ifdef PRECISION_FLOAT
    cublasSnrm2(handle, L, d_result, 1, &norms[i]);
    #elif PRECISION_DOUBLE
    cublasDnrm2(handle, L, d_result, 1, &norms[i]);
    #endif

    if (norms[i] < tol) {
        std::cout << "Break at iteration " << i + 1 << std::endl;
        m=i;
        break;
    } else {
        DevicePrecision inv_eval = 1.0 / norms[i];
        #ifdef PRECISION_FLOAT
        cublasSscal(handle, N, &inv_eval, d_result, 1);
        #elif PRECISION_DOUBLE
        cublasDscal(handle, N, &inv_eval, d_result, 1);
        #endif
        cudaMemcpyChecked(&d_evecs[(i + 1) * N], d_result, N * PRECISION_SIZE, cudaMemcpyDeviceToDevice);
        cudaMemcpyChecked(d_y, d_result, N * PRECISION_SIZE, cudaMemcpyDeviceToDevice);
        m=i;
    }
}


MatrixColMajor Q = MatrixColMajor(N, m+2);
MatrixColMajor H_tilde = MatrixColMajor(max_iters+1, max_iters);

cudaMemcpyChecked(Q.data(), d_evecs, (max_iters+1) * N * PRECISION_SIZE, cudaMemcpyDeviceToHost);
cudaMemcpyChecked(H_tilde.data(), d_h, (max_iters+1) * max_iters * PRECISION_SIZE, cudaMemcpyDeviceToHost);
for (int j = 0; j < m+1; ++j) {H_tilde(j+1, j) = norms[j];} // Insert norms back into Hessenberg diagonal

// Free device memory
cudaFree(d_evecs);
cudaFree(d_proj);
cudaFree(d_y);
cudaFree(d_M);
cudaFree(d_result);
cudaFree(d_h);

// Destroy CUBLAS handle
cublasDestroy(handle);

return ArnoldiPair(Q, H_tilde, m+1);

}



//PARAMETERS:
// const MatrixType& M: Matrix to compute Ritz pairs on
// const size_t& max_iters: Maximum number of basis vectors to compute
// const size_t& basis_size: Number of ritzpairs to return
// const HostPrecison& tol: Breakpoint for ritzvalue size
template <typename MatrixType>
RealEigenPairs<MatrixType> computeRitzPairs(const MatrixType& M, const size_t& max_iters, const size_t& basis_size, const HostPrecision& tol=1e-5) {
    size_t C = 0; // Number of columns in M
    size_t R = 0; // Number of rows in M
    std::tie(R, C) = shape(M);
    // Step 1: Perform Arnoldi iteration to get Q and H_tilde
    ArnoldiPair arnoldiResult = arnoldiEigen(M, max_iters, tol);
    const MatrixColMajor& Q = arnoldiResult.Q;
    const size_t& m = arnoldiResult.m;
    MatrixColMajor H_square = arnoldiResult.H_tilde.block(0, 0, m, m);

    // Sorted evals and corresponding evecs
    RealEigenPairs<MatrixType> H_eigensolution = eigenSolver<MatrixColMajor>(H_square);

    const Vector& eigenvalues = H_eigensolution.values;
    const MatrixType& eigenvectors = H_eigensolution.vectors;
    const size_t& num_eigen_pairs = H_eigensolution.num_pairs;

    // Allocate memory on the device for Q, eigenvectors_real, and the result
    if (Q.size() > 1e7) {    
        DevicePrecision* d_Q = cudaMallocChecked<DevicePrecision>(Q.size() * PRECISION_SIZE);
        DevicePrecision* d_evecs = cudaMallocChecked<DevicePrecision>(eigenvectors.size() * PRECISION_SIZE);
        DevicePrecision* d_RitzVectors = cudaMallocChecked<DevicePrecision>(Q.rows() * num_eigen_pairs * PRECISION_SIZE);

        // Copy Q and eigenvectors_real to the device
        cudaMemcpyChecked(d_Q, Q.data(), Q.size() * PRECISION_SIZE, cudaMemcpyHostToDevice);
        cudaMemcpyChecked(d_evecs, eigenvectors.data(), eigenvectors.size() * PRECISION_SIZE, cudaMemcpyHostToDevice);

        // Create CUBLAS handle
        cublasHandle_t handle;
        cublasCreate(&handle);

        cudaFree(d_Q);
        cudaFree(d_evecs);
        cudaFree(d_RitzVectors);
        cublasDestroy(handle);

        // Add matmul internal once the Mat/Mat operation is implemented
    }

    MatrixType RitzVectors = Q.block(0, 0, m, m) * ((MatrixType::IsRowMajor) ? eigenvectors.transpose() : eigenvectors);
    // Copy the result back to the host
    // MatrixColMajor RitzVectors(Q.rows(), eigenvectors_real.cols());
    // cudaMemcpyChecked(RitzVectors.data(), d_RitzVectors, RitzVectors.size() * PRECISION_SIZE, cudaMemcpyDeviceToHost);

    // Step 5: Free device memory and destroy CUBLAS handle


    // // Return Ritz values (eigenvalues) and Ritz vectors (the result of Q * real(eigenvectors))
    return {eigenvalues, RitzVectors, m};
}

#endif // ARNOLDI_HPP