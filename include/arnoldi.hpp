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

constexpr size_t MAX_EVEC_ON_DEVICE = 1e4;

struct KrylovPair { MatrixColMajor Q; MatrixColMajor H_tilde; size_t m;};

template <typename MatrixType>
KrylovPair krylovIter(const MatrixType& M, const size_t& max_iters, const HostPrecision& tol) {
    size_t N, L; //N=num_rows, L=num_cols
    std::tie(N, L) = shape(M);
    const size_t& num_iters = std::min(max_iters, N - 1);
    Vector v0 = randVecGen(N);
    Vector h_evals(num_iters);
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
    const size_t NUM_EVECS_ON_DEVICE = std::min(MAX_EVEC_ON_DEVICE, num_iters + 1);
    size_t ROWS = MAX_ROW_ALLOC(free_mem, N);

    size_t m = 1;


    // Allocations
// Allocations
DevicePrecision* d_evecs = cudaMallocChecked<DevicePrecision>(NUM_EVECS_ON_DEVICE * N * PRECISION_SIZE);
DevicePrecision* d_proj = cudaMallocChecked<DevicePrecision>(NUM_EVECS_ON_DEVICE * PRECISION_SIZE);
DevicePrecision* d_y = cudaMallocChecked<DevicePrecision>(N * PRECISION_SIZE);
DevicePrecision* d_M = cudaMallocChecked<DevicePrecision>(ROWS * N * PRECISION_SIZE);
DevicePrecision* d_result = cudaMallocChecked<DevicePrecision>(N * PRECISION_SIZE);  // Changed L to N
DevicePrecision* d_h = cudaMallocChecked<DevicePrecision>((num_iters + 1) * num_iters * PRECISION_SIZE);  // Corrected size

Vector norms(num_iters);

// Initial setup
cudaMemcpyChecked(d_y, v0.data(), N * PRECISION_SIZE, cudaMemcpyHostToDevice);
cudaMemcpyChecked(d_evecs, v0.data(), N * PRECISION_SIZE, cudaMemcpyHostToDevice);

cublasHandle_t handle;
cublasCreate(&handle);

constexpr DevicePrecision alpha = 1.0;
constexpr DevicePrecision beta = 0.0;
constexpr DevicePrecision neg_one = -1.0;

Vector temp(N);



for (int i = 0; i < num_iters; i++) {  // Changed to start from 0
    matmul_internal<MatrixType>(M, d_M, d_y, d_result, ROWS, N, L, handle);

    //Modified Gram-Schmidt
    for (int j = 0; j <= i; j++) {
        cublasGemv(handle, CUBLAS_OP_T, N, 1, &alpha, 
                &d_evecs[j * N], 1, d_result, 1, 
                &beta, &d_h[i * (num_iters + 1) + j], 1);
        
        cublasGemv(handle, CUBLAS_OP_N, N, 1, &neg_one, 
                &d_evecs[j * N], 1, &d_h[i * (num_iters + 1) + j], 1, 
                &alpha, d_result, 1);
    }

    MatrixType h_res(num_iters + 1, num_iters);
    cudaMemcpyChecked(h_res.data(), d_h,  (num_iters + 1) * num_iters * sizeof(HostPrecision), cudaMemcpyDeviceToHost);
    // print(h_res);

    // TO-DO: FUCNTION PTR THIS BAD BOY
    #ifdef PRECISION_FLOAT
    cublasSnrm2(handle, L, d_result, 1, &norms[i]);
    #elif PRECISION_DOUBLE
    cublasDnrm2(handle, L, d_result, 1, &norms[i]);
    #endif

    DevicePrecision inv_eval = 1.0 / norms[i];
    #ifdef PRECISION_FLOAT
    cublasSscal(handle, N, &inv_eval, d_result, 1);
    #elif PRECISION_DOUBLE
    cublasDscal(handle, N, &inv_eval, d_result, 1);
    #endif
    cudaMemcpyChecked(&d_evecs[(i + 1) * N], d_result, N * PRECISION_SIZE, cudaMemcpyDeviceToDevice);
    cudaMemcpyChecked(d_y, d_result, N * PRECISION_SIZE, cudaMemcpyDeviceToDevice);

    if (norms[i] < tol) {
        m++;
        break;
    } else {
        m++;
    }
}

m-=1;
MatrixColMajor Q = MatrixColMajor(N, m+1);
MatrixColMajor H_tilde = MatrixColMajor(m+1, m);

cudaMemcpyChecked(Q.data(), d_evecs, (m+1) * N * PRECISION_SIZE, cudaMemcpyDeviceToHost);
cudaMemcpyChecked(H_tilde.data(), d_h, (m+1) * m * PRECISION_SIZE, cudaMemcpyDeviceToHost);
for (int j = 0; j < m; ++j) {H_tilde(j+1, j) = norms[j];} // Insert norms back into Hessenberg diagonal

// Free device memory
cudaFree(d_evecs);
cudaFree(d_proj);
cudaFree(d_y);
cudaFree(d_M);
cudaFree(d_result);
cudaFree(d_h);

std::cout << m << std::endl;

// Destroy CUBLAS handle
cublasDestroy(handle);

return KrylovPair(Q, H_tilde, m);

}


//Function to run N-step Arnoldi Iteration and return only the real ritz-pairs from matrix M
//PARAMETERS:
// const MatrixType& M: Matrix to compute Ritz pairs on
// const size_t& max_iters: Maximum number of basis vectors to compute
// const size_t& basis_size: Number of ritzpairs to return
// const HostPrecison& tol: Breakpoint for ritzvalue size
template <typename MatrixType>
RealEigenPairs<MatrixType> ArnoldiRitzPairs(const MatrixType& M, const size_t& max_iters, const size_t& basis_size, const HostPrecision& tol=1e-5) {

    size_t C = 0; // Number of columns in M
    size_t R = 0; // Number of rows in M
    std::tie(R, C) = shape(M);
    // Step 1: Perform Arnoldi iteration to get Q and H_tilde
    KrylovPair krylovResult = krylovIter(M, max_iters, tol);
    const MatrixColMajor& Q = krylovResult.Q;
    const size_t& m = krylovResult.m;

    MatrixColMajor H_square = krylovResult.H_tilde.block(0, 0, m, m);
    RealEigenPairs<MatrixColMajor> H_eigensolution = eigenSolver<MatrixColMajor>(H_square); //Returns sorted purely real eigenpairs


    const size_t num_eigen_pairs = std::min(H_eigensolution.num_pairs, basis_size);
    const Vector& eigenvalues = (num_eigen_pairs < H_eigensolution.num_pairs) ? H_eigensolution.values.head(num_eigen_pairs) : H_eigensolution.values;
    const MatrixColMajor& eigenvectors = (num_eigen_pairs < H_eigensolution.num_pairs) ? H_eigensolution.vectors.block(0, 0, m, num_eigen_pairs) : H_eigensolution.vectors;

    // Allocate memory on the device for Q, eigenvectors_real, and the result
    #ifdef CUBLAS_MATMUL
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
    #else
    MatrixType RitzVectors;
    MatrixColMajor productHolder = Q.block(0,0,R, m) * eigenvectors;

    if constexpr (MatrixType::IsRowMajor) {RitzVectors = productHolder.transpose();}
    else {RitzVectors = productHolder;}
    #endif

    return {eigenvalues, RitzVectors, num_eigen_pairs};

}

#endif // ARNOLDI_HPP