#ifndef ARNOLDI_HPP
#define ARNOLDI_HPP

#include <array>
#include <tuple>
#include <iostream>
#include <vector>

#include "matmul.hpp"
#include "structs.hpp"
#include "cuda_manager.hpp"
#include "eigenSolver.hpp"

#define DEBUG_ARNOLDI

constexpr size_t MAX_EVEC_ON_DEVICE = 1e4;


// Specialization for real matrices
template <typename T>
struct KrylovTraits<T, std::enable_if_t<!std::is_same_v<typename T::Scalar, ComplexType>>> {
    using InputMatrixType = T;
    using HostScalarType = HostPrecision;
    using DeviceScalarType = DevicePrecision;
    using VectorType = Vector;
    using OutputMatrixType = Matrix;
    using ReturnType = KrylovPair<T>;
    
    static constexpr bool is_complex = false;

    static VectorType initial_vector(size_t N) {
        return randVecGen(N);
    }

    static void normalize_vector(VectorType& v) {
        #ifdef USE_EIGEN
            v.normalize();
        #else
            auto norm_val = norm(v);
            for (auto& val : v) { val /= norm_val; }
        #endif
    }

    static OutputMatrixType convert(const OutputMatrixType& mat) { return mat; }
    static OutputMatrixType convert(const ComplexMatrix& mat) { return mat.real(); }
};

// Specialization for complex matrices
template <typename T>
struct KrylovTraits<T, std::enable_if_t<std::is_same_v<typename T::Scalar, ComplexType>>> {
    using InputMatrixType = T;
    using HostScalarType = ComplexType;
    using DeviceScalarType = DeviceComplexType;
    using VectorType = ComplexVector;
    using OutputMatrixType = ComplexMatrix;
    using ReturnType = KrylovPair<T>;

    static constexpr bool is_complex = true;

    static VectorType initial_vector(size_t N) {
        return randVecGen(N);
    }

    static void normalize_vector(VectorType& v) {
            #ifdef USE_EIGEN
            v.normalize();
        #else
            auto norm_val = norm(v);
            for (auto& val : v) { val /= norm_val; }
        #endif
    }

    static OutputMatrixType convert(const OutputMatrixType& mat) { return mat; }
    static OutputMatrixType convert(const Matrix& mat) { return mat.cast<ComplexType>(); }
};

// Main function for RealKrylovIter using traits
template <typename T>
typename KrylovTraits<T>::ReturnType 
KrylovIter(const T& M, const size_t& max_iters, 
               cublasHandle_t& handle, const HostPrecision& tol = default_tol) {
    using Traits = KrylovTraits<T>;
    using DeviceScalarType = typename Traits::DeviceScalarType;
    using VectorType = typename Traits::VectorType;
    using OutputMatrixType = typename Traits::OutputMatrixType;
    using InputMatrixType = typename Traits::InputMatrixType;
    
    const HostPrecision matnorm = M.norm();
    const size_t SCALAR_SIZE = sizeof(DeviceScalarType);
    size_t N, L;
    #ifdef USE_EIGEN
        N = M.rows();
        L = M.cols();
    #else
        N = rows(M);
        L = cols(M);
    #endif
    assert(max_iters < N && "max_iters must be leq than leading dimension of M");
    
    Vector norms(max_iters);
    VectorType v0 = Traits::initial_vector(N);
    Traits::normalize_vector(v0);
    
    OutputMatrixType Q = OutputMatrixType(N, max_iters + 1);
    OutputMatrixType H_tilde = OutputMatrixType(max_iters + 1, max_iters);
    
    const size_t NUM_EVECS_ON_DEVICE = max_iters + 1; 
    const size_t ROWS = DYNAMIC_ROW_ALLOC(N);
    const size_t DEV_PREC_SIZE = sizeof(DeviceScalarType);
    size_t m = 0;

    DeviceScalarType* d_evecs = cudaMallocChecked<DeviceScalarType>(NUM_EVECS_ON_DEVICE * N * DEV_PREC_SIZE);
    DeviceScalarType* d_proj = cudaMallocChecked<DeviceScalarType>(NUM_EVECS_ON_DEVICE * DEV_PREC_SIZE);
    DeviceScalarType* d_y = cudaMallocChecked<DeviceScalarType>(N * DEV_PREC_SIZE);
    DeviceScalarType* d_M = cudaMallocChecked<DeviceScalarType>(ROWS * N * DEV_PREC_SIZE);
    DeviceScalarType* d_result = cudaMallocChecked<DeviceScalarType>(N * DEV_PREC_SIZE);
    DeviceScalarType* d_h = cudaMallocChecked<DeviceScalarType>((max_iters + 1) * max_iters * DEV_PREC_SIZE);

    cudaMemcpyChecked(d_y, v0.data(), N * DEV_PREC_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpyChecked(d_evecs, v0.data(), N * DEV_PREC_SIZE, cudaMemcpyHostToDevice);

    constexpr DevicePrecision alpha = 1.0;
    constexpr DevicePrecision beta = 0.0;
    constexpr DevicePrecision neg_one = -1.0;

    Vector temp(N);

    // if constexpr (Traits::is_complex) {inv_eval = make_cuDoubleComplex(1.0 / norms[i], 0.0);}
    // else {inv_eval = 1.0 / norms[i];}

    for (int i = 0; i < max_iters; i++) {
        matmul_internal<InputMatrixType>(M, d_M, d_y, d_result, ROWS, N, L, handle);
        cublas::MGS<DeviceScalarType>(handle, d_evecs, d_h, d_result, N, max_iters, i);

        OutputMatrixType h_res(max_iters + 1, max_iters);
        cudaMemcpyChecked(h_res.data(), d_h, (max_iters + 1) * max_iters * sizeof(HostPrecision), cudaMemcpyDeviceToHost);

        cublas::norm<DeviceScalarType>(handle, L, d_result, 1, &norms[i]);
        DevicePrecision inv_eval = 1.0 / norms[i];
        cublas::scale<DeviceScalarType>(handle, N, &inv_eval, d_result, 1);

        cudaMemcpyChecked(&d_evecs[(i + 1) * N], d_result, N * DEV_PREC_SIZE, cudaMemcpyDeviceToDevice);
        cudaMemcpyChecked(d_y, d_result, N * DEV_PREC_SIZE, cudaMemcpyDeviceToDevice);

        if (norms[i] < tol * matnorm) {
            m++;
            break;
        } else {
            m++;
        }
    }

    cudaMemcpyChecked(Q.data(), d_evecs, m * N * DEV_PREC_SIZE, cudaMemcpyDeviceToHost);
    assert(isOrthonormal<OutputMatrixType>(Q.block(0, 0, N, m)));
    cudaMemcpyChecked(H_tilde.data(), d_h, m * (m+1) * DEV_PREC_SIZE, cudaMemcpyDeviceToHost);
    for (int j = 0; j < m ; ++j) { H_tilde(j + 1, j) = norms[j]; } // Insert norms back into Hessenberg diagonal

    // Free device memory
    cudaFree(d_evecs);
    cudaFree(d_proj);
    cudaFree(d_y);
    cudaFree(d_M);
    cudaFree(d_result);
    cudaFree(d_h);

    return typename Traits::ReturnType{
        Q.block(0, 0, N, max_iters), 
        H_tilde.block(0, 0, max_iters, max_iters), 
        m
    };
}

template <typename T>
EigenPairs NaiveArnoldi(const T& M, const size_t& max_iters, 
                           cublasHandle_t& handle, const HostPrecision& tol = 1e-5) {
    using Traits = KrylovTraits<T>;
    using OutputMatrixType = typename Traits::OutputMatrixType;

    size_t R, C;
    std::tie(R, C) = shape(M);

    auto krylovResult = RealKrylovIter(M, max_iters, handle);
    const OutputMatrixType& Q = krylovResult.Q;
    const size_t& m = krylovResult.m;

    OutputMatrixType H_square = krylovResult.H;
    EigenPairs H_eigensolution{};

    #ifdef DEBUG_ARNOLDI
    assert(isOrthonormal<OutputMatrixType>(Q));
    assert(isHessenberg<OutputMatrixType>(H_square));
    #endif

    eigsolver<MatrixColMajor>(H_square, H_eigensolution, m, matrix_type::HESSENBERG);

    const ComplexVector& eigenvalues = H_eigensolution.values;
    const ComplexMatrix& H_EigenVectors = H_eigensolution.vectors;

    return {eigenvalues, Q.block(0, 0, R, m) * H_EigenVectors, false, false, m};
}

#endif // ARNOLDI_HPP