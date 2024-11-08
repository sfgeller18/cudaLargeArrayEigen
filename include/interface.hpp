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
template <typename M, size_t T, size_t N, size_t S, matrix_type Type>
class IRAMEigen {
public:
    struct Traits {
        using OM = typename KrylovTraits<M>::OutputMatrixType;
        using Scalar = typename KrylovTraits<M>::VectorType::Scalar;
        static constexpr bool isComplex = std::is_same_v<Scalar, ComplexType>;
        static constexpr bool isSA = Type == matrix_type::SELFADJOINT;
        using DeviceAllocType = std::conditional<!isComplex & isSA, DevicePrecision, DeviceComplexType>;
        using EigenValueType = std::conditional<isSA, Vector, ComplexVector>;
        using EigenVectorType = std::conditional<isSA & !isComplex, Matrix, ComplexMatrix>;
        static constexpr size_t DEV_PREC_SIZE = sizeof(DeviceAllocType);
    };
public:
    IRAMEigen(const M& Mat, size_t top_k_pairs)
        : M_(Mat), top_k_pairs_(top_k_pairs) {
        using D = Traits::DeviceAllocType;
        cublasCreate(&handle_);
        cusolverDnCreate(&solver_handle_);

        // Get dimensions of M
        N_ = M_.rows();
        L_ = M_.cols();

        //Memory Handlers
        const size_t NUM_EVECS_ON_DEVICE = N_ + 1; // For now just store all evecs on device, shouldn't need to be batched for IRAM (unless mat size  O(1e6))
        const size_t ROWS_ALLOC = DYNAMIC_ROW_ALLOC(N_);

        D* d_evecs = cudaMallocChecked<D>(NUM_EVECS_ON_DEVICE * N_ * Traits::DEV_PREC_SIZE);
        D* d_proj = cudaMallocChecked<D>(NUM_EVECS_ON_DEVICE * Traits::DEV_PREC_SIZE);
        D* d_y = cudaMallocChecked<D>(N_ * Traits::DEV_PREC_SIZE);
        D* d_M = cudaMallocChecked<D>(ROWS_ALLOC * N_ * Traits::DEV_PREC_SIZE);
        D* d_result = cudaMallocChecked<D>(N_ * Traits::DEV_PREC_SIZE);
        D* d_h = cudaMallocChecked<D>((N + 1) * N * Traits::DEV_PREC_SIZE);

        #ifdef CUBLAS_RESTART // Restart requires complex types as shift computed from eigenvalues of Hessenberg operator
        DeviceComplexType* d_Q = cudaMallocChecked<DeviceComplexType>((batch_count + 1) * N_squared * sizeof(DeviceComplexType));
        std::vector<DeviceComplexType*> h_Tauarray(batch_count);
        DeviceComplexType* d_tau = cudaMallocChecked<DeviceComplexType>(batch_count * N * sizeof(DeviceComplexType));    
        for (size_t i = 0; i < N - S; ++i) {h_Tauarray[i] = d_tau + i * N;}
        #endif    

        // Allocate memory for Q and H
        q_h_.Q = ComplexMatrix::Zero(M_.rows(), S);
        q_h_.H = ComplexMatrix::Zero(S, S);

        get_ritzPairs();

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

    }

    ~IRAMEigen() {
        // Cleanup cuBLAS and cuSOLVER handles
        cublasDestroy(handle_);
        cusolverDnDestroy(solver_handle_);


    }

    inline size_t getTopKPairs() const { return top_k_pairs_; }
    inline void setTopKPairs(size_t top_k_pairs) { top_k_pairs_ = top_k_pairs; }
    inline M getMatrixCopy() const { return M_; }
    inline M getMatrixRef() const { return Eigen::Map<M>(M_); }

public:

private: // Private Method to get Q and H matrices.
    void get_ritzPairs() {
        typename KrylovTraits<M>::VectorType v0 = KrylovTraits<M>::initial_vector(N_);
        constexpr size_t PREC_SIZE = Traits::DEV_PREC_SIZE;
        KrylovTraits<M>::normalize_vector(v0);
        Vector norms(N_);
        typename Traits::OM Q(N_, N_);
        typename Traits::OM H_tilde(N_ + 1, N_);
        cudaMemcpyChecked(d_y, v0.data(), N_ * PREC_SIZE, cudaMemcpyHostToDevice);
        size_t iters_run = 0;
        size_t last_iter_reduce = top_k_pairs_ > S ? top_k_pairs_ : S;
        
        while(iters_run < T) {
            size_t m = KrylovIterInternal<M>(M_,  isEmptyQH ? 0 : S, N, handle_, default_tol, N_, L_, ROWS_ALLOC,
                                    d_evecs, d_proj, d_y, d_M, 
                                    d_result, d_h, norms);
            if (isEmptyQH) {isEmptyQH = false;}
            cudaMemcpyChecked(Q.data(), d_evecs, m * N * PREC_SIZE, cudaMemcpyDeviceToHost);
            cudaMemcpyChecked(H_tilde.data(), d_h, m * (m+1) * PREC_SIZE, cudaMemcpyDeviceToHost);
            for (int j = 0; j < m ; ++j) {H_tilde(j + 1, j) = norms[j];}
            #ifdef DEBUG_INTERFACE
            assert(isOrthonormal<typename Traits::OM>(Q.block(0, 0, N, m)));
            assert(isHessenberg<typename Traits::OM>(H_tilde.block(0, 0, m, m)));
            #endif
            q_h_ = {Eigen::Map<typename Traits::OM>(Q.block(0, 0, N, m)), Eigen::Map<typename Traits::OM>(H_tilde.block(0, 0, m, m)), m};
            size_t R = iters_run < T - N ? S : last_iter_reduce;
            reduceArnoldiPair<typename Traits::OM>(q_h_, R, handle_, solver_handle_, resize_type::ZEROS);
            #ifdef DEBUG_INTERFACE
            assert(isHessenberg<typename Traits::OM>(q_h.H.block(0, 0, S, S)));
            assert(isOrthonormal<typename Traits::OM>(q_h.Q.block(0, 0, N, S), 1e-4));
            #endif
            iters_run += N;
            if (iters_run < T) {cudaMemcpyChecked(d_y, d_evecs, N_ * PREC_SIZE, cudaMemcpyDeviceToDevice);} //Copy first basis vector back into d_y
        }
        EigenPairs H_pairs{};
        eigsolver<typename Traits::OM>(q_h_.H, H_pairs, q_h_.m, matrix_type::HESSENBERG);
        eigenvalues = H_pairs.values;
        eigenvectors = q_h_.Q.block(0,0,N_, top_k_pairs_) * H_pairs.vectors;
    }



private: // Math Variables
    const M M_;
    const matrix_type type_;
    KrylovPair<M> q_h_;
    size_t top_k_pairs_;
    typename Traits::EigenValueType eigenvalues;
    typename Traits::EigenVectorType eigenvectors;

private: // Device Pointers
    void* d_evecs;
    void* d_proj;
    void* d_y;
    void* d_M;
    void* d_result;
    void* d_h;
    cublasHandle_t handle_;
    cusolverDnHandle_t solver_handle_;
    
private: // Memory Handling Utils
    size_t N_, L_; // Rows and Columns of M respectively
    const size_t NUM_EVECS_ON_DEVICE = N + 1;
    const size_t ROWS_ALLOC;
    bool isEmptyQH = true;

};

#endif // ARNOLDI_EIGEN_HPP