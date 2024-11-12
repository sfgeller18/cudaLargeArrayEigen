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

template <typename M, typename Enable = void>
struct BasisTraits;

// Specialization for HostPrecision
template <typename M>
struct BasisTraits<M, std::enable_if_t<std::is_same_v<typename M::Scalar, HostPrecision>>> {
    using S = HostPrecision;
    using DS = DevicePrecision;
    using V = Vector;
    using OM = Matrix;
    constexpr static size_t ALLOC_SIZE = sizeof(DS);
};

// Specialization for ComplexType
template <typename M>
struct BasisTraits<M, std::enable_if_t<std::is_same_v<typename M::Scalar, ComplexType>>> {
    using S = ComplexType;
    using DS = DeviceComplexType;
    using V = ComplexVector;
    using OM = ComplexMatrix;
    constexpr static size_t ALLOC_SIZE = sizeof(DS);
};


template <typename M>
struct KrylovPair {
    using MT = BasisTraits<M>::OM;
    MT Q;
    MT H;
    size_t m;

    KrylovPair(const Matrix& q, const Matrix& h, size_t m)
        : Q(q), H(h), m(m) {}

    // Default constructor
    KrylovPair() : m(0) {}
};

template <typename M, typename DS>
int KrylovIterInternal(typename BasisTraits<M>::OM& Q, typename BasisTraits<M>::OM& H_tilde, size_t& m, const M& M_,
                        DS* d_M, DS* d_y, DS* d_result, DS* d_evecs, DS* d_h, Vector& norms, const size_t& num_iters, const size_t& N, const size_t& L, const size_t& ROWS, cublasHandle_t& handle, const HostPrecision& matnorm = 1, const HostPrecision& tol = 1e-5) {
        constexpr size_t ALLOC_SIZE = BasisTraits<M>::ALLOC_SIZE;
        for (int i = 0; i < num_iters; i++) {

        matmul_internal<M, DS>(M_, d_M, d_y, d_result, ROWS, N, L, handle);

        cublas::MGS<DS>(handle, d_evecs, d_h, d_result, N, num_iters, i);
        cublas::norm<DS>(handle, L, d_result, 1, &norms[i]);
        DevicePrecision inv_eval = 1.0 / norms[i];
        cublas::scale<DS>(handle, N, &inv_eval, d_result, 1);

        //Device to Device Memcpys
        cudaMemcpyChecked(&d_evecs[(i + 1) * N], d_result, N * ALLOC_SIZE, cudaMemcpyDeviceToDevice);
        cudaMemcpyChecked(d_y, d_result, N * ALLOC_SIZE, cudaMemcpyDeviceToDevice);

        if (norms[i] < tol * matnorm) {
            m++;
            break;
        } else {
            m++;
        }
    }

    m -= 1;

    cudaMemcpyChecked(Q.data(), d_evecs, (m + 1) * N * PRECISION_SIZE, cudaMemcpyDeviceToHost);
    cudaMemcpyChecked(H_tilde.data(), d_h, (m + 1) * m * PRECISION_SIZE, cudaMemcpyDeviceToHost);
    for (int j = 0; j < m; ++j) { H_tilde(j + 1, j) = norms[j]; } // Insert norms back into Hessenberg diagonal

    return 0;
}


template <typename M>
KrylovPair<M> KrylovIter(const M& M_, const size_t& max_iters, cublasHandle_t& handle, const HostPrecision& tol = default_tol) {
    using S = typename BasisTraits<M>::S;
    using DS = typename BasisTraits<M>::DS;
    using V = typename BasisTraits<M>::V;
    using OM = typename BasisTraits<M>::OM;
    constexpr size_t ALLOC_SIZE = BasisTraits<M>::ALLOC_SIZE;
    const HostPrecision matnorm = M_.norm();
    size_t N, L; // N=num_rows, L=num_cols
    #ifdef USE_EIGEN
        L = M_.cols();
        N = M_.rows();
    #else
        L = cols(M_);
        N = rows(M_);
    #endif

    OM Q(N, max_iters + 1);
    OM H_tilde(max_iters + 1, max_iters);

    assert(max_iters < N && "max_iters must be leq than leading dimension of M");
    const size_t& num_iters = max_iters;
    Vector norms(num_iters);
    V v0 = randVecGen(N);

    #ifdef USE_EIGEN
        v0.normalize();
    #else
        HostPrecision Norm = norm(v0);
        for (HostPrecision& v : v0) { v /= Norm; } // v0 is a norm 1 random vector
    #endif
    size_t m = 1;
    const size_t NUM_EVECS_ON_DEVICE = num_iters + 1; 
    const size_t ROWS = DYNAMIC_ROW_ALLOC(N);

    // CUDA Allocations
    DS* d_evecs = cudaMallocChecked<DS>(NUM_EVECS_ON_DEVICE * N * ALLOC_SIZE);
    DS* d_proj = cudaMallocChecked<DS>(NUM_EVECS_ON_DEVICE * ALLOC_SIZE);
    DS* d_y = cudaMallocChecked<DS>(N * ALLOC_SIZE);
    DS* d_M = cudaMallocChecked<DS>(ROWS * N * ALLOC_SIZE);
    DS* d_result = cudaMallocChecked<DS>(N * ALLOC_SIZE);
    DS* d_h = cudaMallocChecked<DS>((num_iters + 1) * num_iters * ALLOC_SIZE);

    // Initial setup
    cudaMemcpyChecked(d_y, v0.data(), N * ALLOC_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpyChecked(d_evecs, v0.data(), N * ALLOC_SIZE, cudaMemcpyHostToDevice);

    KrylovIterInternal<M, DS>(Q, H_tilde, m, M_, d_M, d_y, d_result, d_evecs, d_h, norms, num_iters, N, L, ROWS, handle, matnorm, tol);


    // Free device memory
    cudaFree(d_evecs);
    cudaFree(d_proj);
    cudaFree(d_y);
    cudaFree(d_M);
    cudaFree(d_result);
    cudaFree(d_h);

    return {Q, H_tilde, m};
}

template <typename M>
EigenPairs NaiveArnoldi(const M& M_, const size_t& max_iters, cublasHandle_t& handle, const HostPrecision& tol = 1e-5) {
    using OM = typename BasisTraits<M>::OM;
    size_t C = 0; // Number of columns in M
    size_t R = 0; // Number of rows in M
    #ifdef USE_EIGEN
        C = M_.cols();
        R = M_.rows();
    #else
        C = cols(M_);
        R = rows(M_);
    #endif

    // Step 1: Perform Arnoldi iteration to get Q and H_tilde
    KrylovPair<M> krylovResult = KrylovIter<M>(M_, max_iters, handle);
    const size_t& m = krylovResult.m;
    const OM& Q = krylovResult.Q.block(0, 0, R, m);
    const OM& H_square = krylovResult.H.block(0, 0, m, m);
    EigenPairs H_eigensolution{};

    eigsolver<OM>(H_square, H_eigensolution, m, matrix_type::HESSENBERG);

    const ComplexVector& eigenvalues = H_eigensolution.values;
    const ComplexMatrix& H_EigenVectors = H_eigensolution.vectors;

    return {eigenvalues, Q * H_EigenVectors, false, false, m};
}


#endif // ARNOLDI_HPP
