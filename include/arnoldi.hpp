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


template <typename S>
struct KrylovPair {
    using MT = std::conditional_t<std::is_same_v<S, HostPrecision>, Matrix, ComplexMatrix>;
    MT Q;
    MT H;
    size_t m;
};

// Internal Logic on Mem Buffers, only possible Memcpy is with matmul. Will handle the small size adequately later but this is as optimal as possible for batched matmuls
template <typename M, typename DS, size_t N, size_t L, size_t num_iters, size_t first_ind = 0>
int KrylovIterInternal(const M& M_, DS* d_M, DS* d_y, DS* d_result, DS* d_evecs, DS* d_h, Vector& norms, const size_t& ROWS, cublasHandle_t& handle, const HostPrecision& matnorm = 1, const HostPrecision& tol = 1e-5) {
        size_t m = 1;
        constexpr size_t ALLOC_SIZE = BasisTraits<M>::ALLOC_SIZE;
        for (int i = 0; i < num_iters - first_ind; i++) {
        matmul_internal<M, DS>(M_, d_M, d_y, d_result, ROWS, N, L, handle);

        cublas::MGS<DS>(handle, d_evecs, d_h, d_result, N, num_iters, i + first_ind);;
        cublas::norm<DS>(handle, L, d_result, 1, &norms[i]);
        DevicePrecision inv_eval = 1.0 / norms[i];
        cublas::scale<DS>(handle, N, &inv_eval, d_result, 1);

        //Device to Device Memcpys
        cudaMemcpyChecked(&d_evecs[(first_ind + i + 1) * N], d_result, N * ALLOC_SIZE, cudaMemcpyDeviceToDevice);
        cudaMemcpyChecked(d_y, d_result, N * ALLOC_SIZE, cudaMemcpyDeviceToDevice);

        if (norms[i] < tol * matnorm) {
            m++;
            break;
        } else {
            m++;
        }
    }

    m -= 1;

    return 0;
}


template <typename M, size_t N, size_t L, size_t max_iters>
KrylovPair<typename M::Scalar> KrylovIter(const M& M_, cublasHandle_t& handle, const HostPrecision& tol = default_tol) {
    using S = typename BasisTraits<M>::S;
    using DS = typename BasisTraits<M>::DS;
    using V = typename BasisTraits<M>::V;
    using OM = typename BasisTraits<M>::OM;
    constexpr size_t ALLOC_SIZE = BasisTraits<M>::ALLOC_SIZE;
    const HostPrecision matnorm = M_.norm();

    OM Q(N, max_iters + 1);
    OM H_tilde(max_iters + 1, max_iters);

    assert(max_iters < N && "max_iters must be leq than leading dimension of M");
    Vector norms(max_iters);
    V v0 = randVecGen<V>(N);

    size_t m = 1;
    const size_t ROWS = DYNAMIC_ROW_ALLOC(N);

    // CUDA Allocations
    DS* d_evecs = cudaMallocChecked<DS>((max_iters + 1) * N * ALLOC_SIZE);
    DS* d_proj = cudaMallocChecked<DS>((max_iters + 1) * ALLOC_SIZE);
    DS* d_y = cudaMallocChecked<DS>(N * ALLOC_SIZE);
    DS* d_M = cudaMallocChecked<DS>(ROWS * N * ALLOC_SIZE);
    DS* d_result = cudaMallocChecked<DS>(N * ALLOC_SIZE);
    DS* d_h = cudaMallocChecked<DS>((max_iters + 1) * max_iters * ALLOC_SIZE);

    // Initial setup
    cudaMemcpyChecked(d_y, v0.data(), N * ALLOC_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpyChecked(d_evecs, v0.data(), N * ALLOC_SIZE, cudaMemcpyHostToDevice);

    KrylovIterInternal<M, DS, N, L, max_iters>(M_, d_M, d_y, d_result, d_evecs, d_h, norms, ROWS, handle, matnorm, tol);

    cudaMemcpyChecked(Q.data(), d_evecs, (max_iters + 1) * N * ALLOC_SIZE, cudaMemcpyDeviceToHost);
    cudaMemcpyChecked(H_tilde.data(), d_h, (max_iters + 1) * m * ALLOC_SIZE, cudaMemcpyDeviceToHost);
    for (int j = 0; j < max_iters; ++j) { H_tilde(j + 1, j) = norms[j]; } // Insert norms back into Hessenberg diagonal

    
    assert(isOrthonormal<OM>(Q.block(0,0,N,max_iters)));
    assert(isHessenberg<OM>(H_tilde.block(0,0,m, max_iters)));

    // Free device memory
    cudaFree(d_evecs);
    cudaFree(d_proj);
    cudaFree(d_y);
    cudaFree(d_M);
    cudaFree(d_result);
    cudaFree(d_h);

    return {Q, H_tilde, max_iters};
}

template <typename M, size_t N, size_t L, size_t max_iters>
ComplexEigenPairs NaiveArnoldi(const M& M_, cublasHandle_t& handle, const HostPrecision& tol = 1e-5) {
    using OM = typename BasisTraits<M>::OM;

    // Step 1: Perform Arnoldi iteration to get Q and H_tilde
    KrylovPair<typename M::Scalar> krylovResult = KrylovIter<M, N, L, max_iters>(M_, handle);
    const size_t& m = krylovResult.m;
    const OM& Q = krylovResult.Q.block(0, 0, N, m);
    const OM& H_square = krylovResult.H.block(0, 0, m, m);
    ComplexEigenPairs H_eigensolution{};

    eigsolver<OM, matrix_type::HESSENBERG>(H_square, H_eigensolution, m);

    const ComplexVector& eigenvalues = H_eigensolution.values;
    const ComplexMatrix& H_EigenVectors = H_eigensolution.vectors;

    return {eigenvalues, Q * H_EigenVectors, m};
}



#endif // ARNOLDI_HPP