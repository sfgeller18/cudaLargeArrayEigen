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

struct RealKrylovPair;
struct ComplexKrylovPair;


struct ComplexKrylovPair {
    ComplexMatrix Q;
    ComplexMatrix H;
    size_t m;

    ComplexKrylovPair(const RealKrylovPair& realPair);
    ComplexKrylovPair(const ComplexMatrix& q, const ComplexMatrix& h, size_t m)
        : Q(q), H(h), m(m) {}

    // Default constructor
    ComplexKrylovPair() : m(0) {}
};

struct RealKrylovPair {
    MatrixColMajor Q;
    MatrixColMajor H;
    size_t m;

    RealKrylovPair(const ComplexKrylovPair& complexPair);
    RealKrylovPair(const MatrixColMajor& q, const MatrixColMajor& h, size_t m)
        : Q(q), H(h), m(m) {}

    // Default constructor
    RealKrylovPair() : m(0) {}
};

    // Copy constructor from RealKrylovPair
    ComplexKrylovPair::ComplexKrylovPair(const RealKrylovPair& realPair)
        : Q(realPair.Q),
          H(realPair.H),
          m(realPair.m) {}


    // Copy constructor from ComplexKrylovPair
    RealKrylovPair::RealKrylovPair(const ComplexKrylovPair& complexPair)
        : Q(complexPair.Q.real()), // Purging imaginary part
          H(complexPair.H.real()), // Purging imaginary part
          m(complexPair.m) {}


inline void cublasMGS(cublasHandle_t handle, 
                       const HostPrecision* d_evecs, 
                       HostPrecision* d_h, 
                       HostPrecision* d_result, 
                       const HostPrecision alpha, 
                       const HostPrecision neg_one, 
                       int N, 
                       int num_iters, 
                       int i) {
    // Modified Gram-Schmidt
    for (int j = 0; j <= i; j++) {
        cublasGemv(handle, CUBLAS_OP_T, N, 1, &alpha, 
                   &d_evecs[j * N], 1, d_result, 1, 
                   &neg_one, &d_h[i * (num_iters + 1) + j], 1);
        
        cublasGemv(handle, CUBLAS_OP_N, N, 1, &neg_one, 
                   &d_evecs[j * N], 1, &d_h[i * (num_iters + 1) + j], 1, 
                   &alpha, d_result, 1);
    }
}

template <typename MatrixType>
RealKrylovPair RealKrylovIter(const MatrixType& M, const size_t& max_iters, cublasHandle_t& handle, const HostPrecision& tol = 1e-10) {
    const HostPrecision matnorm = M.norm();
    size_t N, L; // N=num_rows, L=num_cols
    std::tie(N, L) = shape(M);

    assert(max_iters < N && "max_iters must be leq than leading dimension of M");
    const size_t& num_iters = max_iters;

    Vector v0 = randVecGen(N);
    Vector h_evals(num_iters);
    Vector norms(num_iters);

#ifdef USE_EIGEN
    h_evals[0] = v0.norm();
    v0.normalize();
#else
    h_evals[0] = norm(v0);
    for (HostPrecision& v : v0) { v /= h_evals[0]; } // v0 is a norm 1 random vector
#endif

    size_t m = 1;
    const size_t NUM_EVECS_ON_DEVICE = num_iters + 1; 
    const size_t ROWS = DYNAMIC_ROW_ALLOC(N);

    // CUDA Allocations
    DevicePrecision* d_evecs = cudaMallocChecked<DevicePrecision>(NUM_EVECS_ON_DEVICE * N * PRECISION_SIZE);
    DevicePrecision* d_proj = cudaMallocChecked<DevicePrecision>(NUM_EVECS_ON_DEVICE * PRECISION_SIZE);
    DevicePrecision* d_y = cudaMallocChecked<DevicePrecision>(N * PRECISION_SIZE);
    DevicePrecision* d_M = cudaMallocChecked<DevicePrecision>(ROWS * N * PRECISION_SIZE);
    DevicePrecision* d_result = cudaMallocChecked<DevicePrecision>(N * PRECISION_SIZE);
    DevicePrecision* d_h = cudaMallocChecked<DevicePrecision>((num_iters + 1) * num_iters * PRECISION_SIZE);

    // Initial setup
    cudaMemcpyChecked(d_y, v0.data(), N * PRECISION_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpyChecked(d_evecs, v0.data(), N * PRECISION_SIZE, cudaMemcpyHostToDevice);

    constexpr DevicePrecision alpha = 1.0;
    constexpr DevicePrecision beta = 0.0;
    constexpr DevicePrecision neg_one = -1.0;

    Vector temp(N);

    for (int i = 0; i < num_iters; i++) {
        matmul_internal<MatrixType>(M, d_M, d_y, d_result, ROWS, N, L, handle);

        // Modified Gram-Schmidt
        cublasMGS(handle, d_evecs, d_h, d_result, alpha, neg_one, N, num_iters, i);

        MatrixType h_res(num_iters + 1, num_iters);
        cudaMemcpyChecked(h_res.data(), d_h, (num_iters + 1) * num_iters * sizeof(HostPrecision), cudaMemcpyDeviceToHost);

        cublasNorm(handle, L, d_result, 1, &norms[i]);
        DevicePrecision inv_eval = 1.0 / norms[i];
        cublasScale(handle, N, &inv_eval, d_result, 1);

        cudaMemcpyChecked(&d_evecs[(i + 1) * N], d_result, N * PRECISION_SIZE, cudaMemcpyDeviceToDevice);
        cudaMemcpyChecked(d_y, d_result, N * PRECISION_SIZE, cudaMemcpyDeviceToDevice);

        if (norms[i] < tol * matnorm) {
            m++;
            break;
        } else {
            m++;
        }
    }

    m -= 1;
    MatrixColMajor Q = MatrixColMajor(N, m + 1);
    MatrixColMajor H_tilde = MatrixColMajor(m + 1, m);

    cudaMemcpyChecked(Q.data(), d_evecs, (m + 1) * N * PRECISION_SIZE, cudaMemcpyDeviceToHost);
    cudaMemcpyChecked(H_tilde.data(), d_h, (m + 1) * m * PRECISION_SIZE, cudaMemcpyDeviceToHost);
    for (int j = 0; j < m; ++j) { H_tilde(j + 1, j) = norms[j]; } // Insert norms back into Hessenberg diagonal

    // Free device memory
    cudaFree(d_evecs);
    cudaFree(d_proj);
    cudaFree(d_y);
    cudaFree(d_M);
    cudaFree(d_result);
    cudaFree(d_h);

    return RealKrylovPair(Q.block(0, 0, N, m), H_tilde.block(0, 0, m, m), m);
}

template <typename MatrixType>
EigenPairs NaiveRealArnoldi(const MatrixType& M, const size_t& max_iters, const size_t& basis_size, const HostPrecision& tol = 1e-5) {
    size_t C = 0; // Number of columns in M
    size_t R = 0; // Number of rows in M
    std::tie(R, C) = shape(M);

    // Step 1: Perform Arnoldi iteration to get Q and H_tilde
    RealKrylovPair krylovResult = RealKrylovIter(M, max_iters, tol);
    const MatrixColMajor& Q = krylovResult.Q;
    const size_t& m = krylovResult.m;

    MatrixColMajor H_square = krylovResult.H;
    EigenPairs H_eigensolution{};

    eigsolver<MatrixColMajor>(H_square, H_eigensolution, m, matrix_type::HESSENBERG);

    const size_t num_eigen_pairs = std::min(H_eigensolution.num_pairs, basis_size);
    const Vector& eigenvalues = (num_eigen_pairs < H_eigensolution.num_pairs) ? H_eigensolution.values.head(num_eigen_pairs) : H_eigensolution.values;
    const MatrixColMajor& H_EigenVectors = (num_eigen_pairs < H_eigensolution.num_pairs) ? H_eigensolution.vectors.block(0, 0, m, num_eigen_pairs) : H_eigensolution.vectors;

    return {eigenvalues, Q.block(0, 0, R, m) * H_EigenVectors, false, false, num_eigen_pairs};
}

#endif // ARNOLDI_HPP
