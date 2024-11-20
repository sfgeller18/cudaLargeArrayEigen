//Main Interface class for Arnoldi Iteration, both Naive and Restarted
#ifndef ARNOLDI_EIGEN_HPP
#define ARNOLDI_EIGEN_HPP

#include "cuda_manager.hpp"
#include "eigenSolver.hpp"
#include "utils.hpp"
#include "structs.hpp"
#include "arnoldi.hpp"
#include "shift.hpp"



//M := Matrix Type, T:= Total Iters, N := Maximum Naive Iters, S := Shift Basis Size
template <typename M, typename D, typename ValueType, typename VectorType, size_t T, size_t N, size_t S, matrix_type Type>
class IRAMEigen {
public:
    IRAMEigen(const M& Mat, size_t top_k_pairs)
        : M_(Mat), top_k_pairs_(top_k_pairs) {
        initializeHandles();
        allocateMemory();
        get_ritzPairs();
        cleanup();
    }

    ~IRAMEigen() {
        cleanup();
    }

    inline size_t getTopKPairs() const { return top_k_pairs_; }
    inline void setTopKPairs(size_t top_k_pairs) { top_k_pairs_ = top_k_pairs; }
    inline M getMatrixCopy() const { return M_; }
    inline M getMatrixRef() const { return Eigen::Map<M>(M_); }
    inline ValueType getEigenvalues() const { return eigenvalues; }
    inline VectorType getEigenvectors() const { return eigenvectors; }

private:
    void initializeHandles() {
        cublasCreate(&handle_);
        cusolverDnCreate(&solver_handle_);
    }

    void allocateMemory() {
        // Get dimensions of M
        N_ = M_.rows();
        L_ = M_.cols();

        const size_t NUM_EVECS_ON_DEVICE = N_ + 1;
        ROWS_ALLOC = DYNAMIC_ROW_ALLOC(N_);
        static constexpr size_t PREC_SIZE = sizeof(D);

        // Allocate device memory for vectors and matrices
        d_evecs = cudaMallocChecked<D>(NUM_EVECS_ON_DEVICE * N_ * PREC_SIZE);
        d_proj = cudaMallocChecked<D>(NUM_EVECS_ON_DEVICE * PREC_SIZE);
        d_y = cudaMallocChecked<D>(N_ * PREC_SIZE);
        d_M = cudaMallocChecked<D>(ROWS_ALLOC * N_ * PREC_SIZE);
        d_result = cudaMallocChecked<D>(N_ * PREC_SIZE);
        d_h = cudaMallocChecked<D>((N + 1) * N * PREC_SIZE);

        #ifdef CUBLAS_RESTART
        d_Q = cudaMallocChecked<DeviceComplexType>((batch_count + 1) * N_squared * sizeof(DeviceComplexType));
        std::vector<DeviceComplexType*> h_Tauarray(batch_count);
        d_tau = cudaMallocChecked<DeviceComplexType>(batch_count * N * sizeof(DeviceComplexType));
        for (size_t i = 0; i < N - S; ++i) { h_Tauarray[i] = d_tau + i * N; }
        #endif    

        q_h_.Q = ComplexMatrix::Zero(M_.rows(), S);
        q_h_.H = ComplexMatrix::Zero(S, S);
    }

    void cleanup() {
        cudaFreeChecked(d_evecs);
        cudaFreeChecked(d_proj);
        cudaFreeChecked(d_y);
        cudaFreeChecked(d_M);
        cudaFreeChecked(d_result);
        cudaFreeChecked(d_h);

        #ifdef CUBLAS_RESTART
        cudaFreeChecked(d_Q);
        cudaFreeChecked(d_tau);
        #endif

        cublasDestroy(handle_);
        cusolverDnDestroy(solver_handle_);
    }

    void get_ritzPairs() {
        constexpr bool isRealM = std::is_same<typename M::Scalar, HostPrecision>::value;
        using VecType = std::conditional<isRealM, Vector, ComplexVector>::type;
        using OM = std::conditional<isRealM, Matrix, ComplexMatrix>::type;
        VecType v0 = VecType::Random(N_);
        Vector norms(N_);
        v0.normalize(); 
        constexpr size_t PREC_SIZE = sizeof(D);
        OM Q(N_, N_);
        OM H_tilde(N_ + 1, N_);
        cudaMemcpyChecked(d_y, v0.data(), N_ * PREC_SIZE, cudaMemcpyHostToDevice);
        
        size_t iters_run = 0;
        size_t last_iter_reduce = top_k_pairs_ > S ? top_k_pairs_ : S;
        
        while (iters_run < T) {
            size_t m = KrylovIterInternal<M>(M_, S, N, handle_, default_tol, N_, L_, ROWS_ALLOC,
                d_evecs, d_proj, d_y, d_M, d_result, d_h, norms);
            
            if (isEmptyQH) { isEmptyQH = false; }
            
            cudaMemcpyChecked(Q.data(), d_evecs, m * N * PREC_SIZE, cudaMemcpyDeviceToHost);
            cudaMemcpyChecked(H_tilde.data(), d_h, m * (m + 1) * PREC_SIZE, cudaMemcpyDeviceToHost);
            
            for (int j = 0; j < m; ++j) { H_tilde(j + 1, j) = norms[j]; }
            
            #ifdef DEBUG_INTERFACE
            assert(isOrthonormal<typename Traits::OM>(Q.block(0, 0, N, m)));
            assert(isHessenberg<typename Traits::OM>(H_tilde.block(0, 0, m, m)));
            #endif
            
            q_h_ = {std::move(Q.block(0, 0, N, m)),
                    std::move(H_tilde.block(0, 0, m, m)), m};
            
            size_t R = iters_run < T - N ? S : last_iter_reduce;
            reduceArnoldiPair(q_h_, R, handle_, solver_handle_, resize_type::ZEROS);
            
            #ifdef DEBUG_INTERFACE
            assert(isHessenberg<OM>(q_h.H.block(0, 0, S, S)));
            assert(isOrthonormal<OM>(q_h.Q.block(0, 0, N, S), 1e-4));
            #endif
            
            iters_run += N;
            if (iters_run < T) { cudaMemcpyChecked(d_y, d_evecs, N_ * PREC_SIZE, cudaMemcpyDeviceToDevice); }
        }

        EigenPairs H_pairs{}; // Need Complex EigenPairs for Hessenberg H
        eigsolver<ComplexMatrix>(q_h_.H, H_pairs, q_h_.m, matrix_type::HESSENBERG);
        eigenvalues = H_pairs.values.real();
        eigenvectors = (q_h_.Q.block(0, 0, N_, top_k_pairs_) * H_pairs.vectors).real();
    }

private:
    M M_;
    size_t N_, L_;
    size_t top_k_pairs_;
    cublasHandle_t handle_;
    cusolverDnHandle_t solver_handle_;
    ValueType eigenvalues;
    VectorType eigenvectors;
    bool isEmptyQH = true;
    bool isRealM;
    ComplexKrylovPair q_h_;

    size_t ROWS_ALLOC;

    D* d_evecs;
    D* d_proj;
    D* d_y;
    D* d_M;
    D* d_result;
    D* d_h;
    
    #ifdef CUBLAS_RESTART
    DeviceComplexType* d_Q;
    DeviceComplexType* d_tau;
    #endif
};

// Type Specialization for Hermitian Matrices (Real Evals, Complex Evecs)
template <typename M, size_t T, size_t N, size_t S>
class IRAMEigen<M, DeviceComplexType, Vector, ComplexMatrix, T, N, S, matrix_type::SELFADJOINT> {
public:
    struct Traits {
        using OM = ComplexMatrix;
        using Scalar = typename KrylovTraits<M>::VectorType::Scalar;
        using DeviceAllocType = DeviceComplexType;  // Complex precision type
        using EigenVectorType = ComplexMatrix;
        using EigenValueType = Vector;
        static constexpr size_t DEV_PREC_SIZE = sizeof(DeviceComplexType);
    };

    IRAMEigen(const M& Mat, size_t top_k_pairs)
        : IRAMEigen<M, DeviceComplexType, Vector, ComplexMatrix, T, N, S, matrix_type::SELFADJOINT>(Mat, top_k_pairs) {}
};

// Type Specialization for Real Symmetric Matrices (Real Evals, Real Evecs)
template <typename M, size_t T, size_t N, size_t S>
class IRAMEigen<M, DevicePrecision, Vector, Matrix, T, N, S, matrix_type::SELFADJOINT> {
public:
    struct Traits {
        using OM = Matrix;
        using Scalar = typename M::Scalar;
        using DeviceAllocType = DevicePrecision; // Real precision type
        using EigenVectorType = Matrix;
        using EigenValueType = Vector;
        static constexpr size_t DEV_PREC_SIZE = sizeof(DevicePrecision);
    };

    IRAMEigen(const M& Mat, size_t top_k_pairs)
        : IRAMEigen<M, DevicePrecision, Vector, Matrix, T, N, S, matrix_type::SELFADJOINT>(Mat, top_k_pairs) {}
};


// USE IRAM for Complex/Real non-symmetric matrices
template <typename M, size_t T, size_t N, size_t S>
using IRAM = IRAMEigen<M, DeviceComplexType, ComplexVector, ComplexMatrix, T, N, S, matrix_type::REGULAR>;

// USE IRAMHermitian for Hermitian matrices
template <typename M, size_t T, size_t N, size_t S>
using IRAMHermitian = IRAMEigen<M, DeviceComplexType, Vector, ComplexMatrix, T, N, S, matrix_type::SELFADJOINT>;

//USE IRAMSymmetric for Real Symmetric matrices
template <typename M, size_t T, size_t N, size_t S>
using IRAMSymmetric = IRAMEigen<M, DeviceComplexType, Vector, Matrix, T, N, S, matrix_type::SELFADJOINT>;

#endif // ARNOLDI_EIGEN_HPP