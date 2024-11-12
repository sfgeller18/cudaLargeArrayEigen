#ifndef CUDA_MANAGER_HPP
#define CUDA_MANAGER_HPP

    #include <cuda_runtime.h>
    #include <cublas_v2.h>
    #include <cusolverDn.h>

using DeviceComplexType = cuDoubleComplex;

    template <typename T>
constexpr T getOne() {
    if constexpr (std::is_same_v<T, cuDoubleComplex>) {
        return cuDoubleComplex(1.0, 0.0);
    } else if constexpr (std::is_same_v<T, cuComplex>) {
        return make_cuComplex(1.0f, 0.0f);
    } else {
        return T(1.0);
    }
}

template <typename T>
constexpr T getZero() {
    if constexpr (std::is_same_v<T, cuDoubleComplex>) {
        return cuDoubleComplex(0.0, 0.0);
    } else if constexpr (std::is_same_v<T, cuComplex>) {
        return cuDoubleComplex(0.0f, 0.0f);
    } else {
        return T(0.0);
    }
}

template <typename T>
constexpr T getNegOne() {
    if constexpr (std::is_same_v<T, cuDoubleComplex>) {
        return cuDoubleComplex(-1.0, 0.0);
    } else if constexpr (std::is_same_v<T, cuComplex>) {
        return make_cuComplex(-1.0f, 0.0f);
    } else {
        return T(-1.0);
    }
}

    
    enum class CuRetType {
        HOST,
        DEVICE
    };
        
    class CudaError : public std::runtime_error {
    public:
        explicit CudaError(const std::string& message) : std::runtime_error(message) {}
    };

    template <typename T>
    inline T* cudaMallocChecked(size_t size) {
        T* d_ptr = nullptr;
        cudaError_t error = cudaMalloc(&d_ptr, size);
        if (error != cudaSuccess) {throw CudaError("cudaMalloc failed: " + std::string(cudaGetErrorString(error)));}
        return d_ptr;
    }

    inline void cudaMemcpyChecked(void* dst, const void* src, size_t count, cudaMemcpyKind kind) {
        cudaError_t error = cudaMemcpy(dst, src, count, kind);
        if (error != cudaSuccess) {
            throw CudaError("cudaMemcpy failed: " + std::string(cudaGetErrorString(error)));
        }
    }

    inline void cudaFreeChecked(void* d_ptr) {
        if (d_ptr != nullptr) { // Check if the pointer is not null
            cudaError_t error = cudaFree(d_ptr);
            if (error != cudaSuccess) {
                throw CudaError("cudaFree failed: " + std::string(cudaGetErrorString(error)));
            }
        } else {
            throw std::invalid_argument("Pointer passed to cudaFreeChecked is null.");
        }
    }

    constexpr size_t MEM_BUFFER = 500 * (1024*1024); // 500MB Memory Buffer for any given device
    constexpr size_t MAX_ROW_ALLOC(const size_t& num_bytes, const size_t& row_length) {
        return static_cast<size_t>((num_bytes - MEM_BUFFER) / (PRECISION_SIZE * row_length));
    }

    inline size_t DYNAMIC_ROW_ALLOC(const size_t& N) {
        size_t free_mem = 0;
        size_t total_mem = 0;
        cudaError_t error = cudaMemGetInfo(&free_mem, &total_mem);
        return MAX_ROW_ALLOC(free_mem, N);
    }


// Error checking macro
#define CHECK_CUDA(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        printf("CUDA API failed at line %d with error: %s (%d)\n",             \
               __LINE__, cudaGetErrorString(status), status);                  \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

#define CHECK_CUSOLVER(func)                                                   \
{                                                                              \
    cusolverStatus_t status = (func);                                          \
    if (status != CUSOLVER_STATUS_SUCCESS) {                                   \
        printf("cuSOLVER API failed at line %d with error: %d\n",              \
               __LINE__, status);                                              \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

#define CHECK_CUBLAS(func)                                                     \
{                                                                              \
    cublasStatus_t status = (func);                                            \
    if (status != CUBLAS_STATUS_SUCCESS) {                                     \
        printf("cuBLAS API failed at line %d with error: %d\n",                \
               __LINE__, status);                                              \
    }                                                                          \
}

namespace cublas {
    template <typename T>
    struct GemvTraits;

    template <typename T>
    struct GemmTraits;

    template <typename T>
    struct NormTraits;

    template <typename T>
    struct BatchedQRTraits;

    template <typename T>
    struct ScaleTraits;

    // ==================== TRAIT SPECIALIZATIONS ====================

    template <>
    struct GemvTraits<DevicePrecision> {
        #ifdef PRECISION_FLOAT
            static constexpr auto gemvFunc = &cublasSgemv;
        #elif PRECISION_DOUBLE
            static constexpr auto gemvFunc = &cublasDgemv;
        #else
            static_assert(false, "Unsupported type for cublasMGS.");
        #endif
    };

    template <>
    struct GemvTraits<DeviceComplexType> {
        #ifdef PRECISION_FLOAT
            static constexpr auto gemvFunc = &cublasCgemv;
        #elif PRECISION_DOUBLE
            static constexpr auto gemvFunc = &cublasZgemv;
        #else
            static_assert(false, "Unsupported type for cublasMGS.");
        #endif
    };


        template <>
    struct GemmTraits<DevicePrecision> {
        #ifdef PRECISION_FLOAT
            static constexpr auto gemmFunc = &cublasSgemm;
        #elif PRECISION_DOUBLE
            static constexpr auto gemmFunc = &cublasDgemm;
        #else
            static_assert(false, "Unsupported type for cublasMGS.");
        #endif
    };

    template <>
    struct GemmTraits<DeviceComplexType> {
        #ifdef PRECISION_FLOAT
            static constexpr auto gemmFunc = &cublasCgemm;
        #elif PRECISION_DOUBLE
            static constexpr auto gemmFunc = &cublasZgemm;
        #else
            static_assert(false, "Unsupported type for cublasMGS.");
        #endif
    };



    template <>
    struct NormTraits<DevicePrecision> {
        #ifdef PRECISION_FLOAT
            static constexpr auto normFunc = &cublasSnrm2;
        #elif PRECISION_DOUBLE
            static constexpr auto normFunc = &cublasDnrm2;
        #else
            static_assert(false, "Unsupported type for cublasMGS.");
        #endif
    };

    template <>
    struct NormTraits<DeviceComplexType> {
        #ifdef PRECISION_FLOAT
            static constexpr auto normFunc = &cublasScnrm2;
        #elif PRECISION_DOUBLE
            static constexpr auto normFunc = &cublasDznrm2;
        #else
            static_assert(false, "Unsupported type for cublasMGS.");
        #endif
    };

    
    template <>
    struct BatchedQRTraits<DevicePrecision> {
        #ifdef PRECISION_FLOAT
            static constexpr auto qrFunc = &cublasSgeqrfBatched;
        #elif PRECISION_DOUBLE
            static constexpr auto qrFunc = &cublasDgeqrfBatched;
        #else
            static_assert(false, "Unsupported type for cublasMGS.");
        #endif
    };

    template <>
    struct BatchedQRTraits<DeviceComplexType> {
        #ifdef PRECISION_FLOAT
            static constexpr auto qrFunc = &cublasCgeqrfBatched;
        #elif PRECISION_DOUBLE
            static constexpr auto qrFunc = &cublasZgeqrfBatched;
        #else
            static_assert(false, "Unsupported type for cublasMGS.");
        #endif
    };

    



    template <>
    struct ScaleTraits<DeviceComplexType> {
        #ifdef PRECISION_FLOAT
            static constexpr auto scaleFunc = &cublasCsscal;
        #elif PRECISION_DOUBLE
            static constexpr auto scaleFunc = &cublasZdscal;
        #else
            static_assert(false, "Unsupported type for cublasMGS.");
        #endif
    };

    template<>
    struct ScaleTraits<DevicePrecision> {
        #ifdef PRECISION_FLOAT
            static constexpr auto scaleFunc = &cublasSscal;
        #elif PRECISION_DOUBLE
            static constexpr auto scaleFunc = &cublasDscal;
        #else
            static_assert(false, "Unsupported type for cublasMGS.");
        #endif
    };

    // ==================== TRAIT INTERFACES ====================

    template <typename T>
    inline cublasStatus_t gemv(cublasHandle_t handle, cublasOperation_t trans, int m, int n,
                const T* alpha, const T* A, int lda,
                const T* x, int incx, const T* beta,
                T* y, int incy) {
        auto cublasGemv = GemvTraits<T>::gemvFunc;
        return cublasGemv(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
    }

    template <typename T>
    inline cublasStatus_t gemm(cublasHandle_t handle, cublasOperation_t transA, cublasOperation_t transB, 
                            int m, int n, int k, 
                            const T* alpha, const T* A, int lda, 
                            const T* B, int ldb, 
                            const T* beta, T* C, int ldc) {
        auto cublasGemm = GemmTraits<T>::gemmFunc;
        return cublasGemm(handle, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    }

    template <typename T>
    inline cublasStatus_t norm(cublasHandle_t handle, int N, const T* d_result, int incx, HostPrecision* norms) {
        auto cublasNorm = NormTraits<T>::normFunc;
        return cublasNorm(handle, N, d_result, incx, norms);
    }

    template <typename T>
    inline cublasStatus_t batchedQR(cublasHandle_t handle, int m, int n, T** d_Aarray, int lda, T** d_Tauarray, int* d_info, int batch_count) {
        auto cublasQR = BatchedQRTraits<T>::qrFunc;
        return cublasQR(handle, m, n, d_Aarray, lda, d_Tauarray, d_info, batch_count);
    }

    template <typename T>
    inline cublasStatus_t scale(cublasHandle_t handle, int N, const DevicePrecision* alpha, T* x, int incx) {
        auto cublasScale = ScaleTraits<T>::scaleFunc;
        return cublasScale(handle, N, alpha, x, incx);
    }



// ==================== LINALG ROUTINES ====================

template <typename T>
inline void MGS(cublasHandle_t handle,
                     const T* d_evecs,
                     T* d_h,
                     T* d_result,
                     int N,
                     int num_iters,
                     int i) {
    constexpr T NEG_ONE = getNegOne<T>();
    constexpr T ONE = getOne<T>();
    constexpr T ZERO = getZero<T>();
    for (int j = 0; j <= i; j++) {
        // Compute projection coefficient <v_j, w> and store in H(j,i)
        cublas::gemv<T>(handle, CUBLAS_OP_T, N, 1, &ONE,
                   &d_evecs[j * N], N, d_result, 1,
                   &ZERO, &d_h[i * (num_iters + 1) + j], 1);
        
        // Subtract projection: w = w - h_ij * v_j
        cublas::gemv<T>(handle, CUBLAS_OP_N, N, 1, &NEG_ONE,
                   &d_evecs[j * N], N, &d_h[i * (num_iters + 1) + j], 1,
                   &ONE, d_result, 1);
    }
}

} // namespace cublas


#endif // CUDA_MANAGER_HPP