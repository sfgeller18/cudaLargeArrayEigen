#ifndef CUDA_MANAGER_HPP
#define CUDA_MANAGER_HPP

    #include <cuda_runtime.h>
    #include <cublas_v2.h>
    #include <cusolverDn.h>

using DeviceComplexType = cuDoubleComplex;

    
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

#if defined(PRECISION_FLOAT)
    using cublasGemvPtr = cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, int, int,
                                               const float*, const float*, int,
                                               const float*, int, const float*, float*, int);
    using cublasBatchedQRPtr = cublasStatus_t (*)(
        cublasHandle_t handle, int m, int n, cuComplex* Aarray[], int lda, cuComplex* Tauarray[], int* info, int batchSize);
    using cublasNormPtr = cublasStatus_t (*)(cublasHandle_t, int, const float*, int, float*);
    using cublasScalePtr = cublasStatus_t (*)(cublasHandle_t, int, const float*, float*, int);
#elif defined(PRECISION_DOUBLE)
    using cublasGemvPtr = cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, int, int,
                                               const double*, const double*, int,
                                               const double*, int, const double*, double*, int);
    using cublasBatchedQRPtr = cublasStatus_t (*)(
        cublasHandle_t handle, int m, int n, cuDoubleComplex* const Aarray[], int lda, cuDoubleComplex* const Tauarray[], int* info, int batchSize);
    using cublasNormPtr = cublasStatus_t (*)(cublasHandle_t, int, const double*, int, double*);
    using cublasScalePtr = cublasStatus_t (*)(cublasHandle_t, int, const double*, double*, int);
#elif defined(PRECISION_FLOAT16)
    using cublasGemvPtr = cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, int, int,
                                               const __half*, const __half*, int,
                                               const __half*, int, const __half*, __half*, int);
    using cublasBatchedQRPtr = void*; // No specific function for float16
    using cublasNormPtr = void*; // No specific function for float16
    using cublasScalePtr = void*; // No specific function for float16
#else
    #error "No precision defined! Please define PRECISION_FLOAT, PRECISION_DOUBLE, or PRECISION_FLOAT16."
#endif

// Function to return the correct normalization function pointer based on precision
constexpr cublasNormPtr getNormFunction() {
#if defined(PRECISION_FLOAT)
    return cublasSnrm2;
#elif defined(PRECISION_DOUBLE)
    return cublasDnrm2;
#elif defined(PRECISION_FLOAT16)
    return nullptr; // No normalization function for float16
#else
    return nullptr; // Should never reach here due to the preprocessor error
#endif
}

// Function to return the correct scaling function pointer based on precision
constexpr cublasScalePtr getScaleFunction() {
#if defined(PRECISION_FLOAT)
    return cublasSscal;
#elif defined(PRECISION_DOUBLE)
    return cublasDscal;
#elif defined(PRECISION_FLOAT16)
    return nullptr; // No scaling function for float16
#else
    return nullptr; // Should never reach here due to the preprocessor error
#endif
}

// Function to return the correct QR function pointer based on precision
constexpr cublasBatchedQRPtr getBatchedQRFunction() {
#if defined(PRECISION_FLOAT)
    return cublasCgeqrfBatched;
#elif defined(PRECISION_DOUBLE)
    return cublasZgeqrfBatched;
#elif defined(PRECISION_FLOAT16)
    return nullptr;
#else
    return nullptr; // Should never reach here due to the preprocessor error
#endif
}

// Function to return the correct gemv function pointer based on precision
constexpr cublasGemvPtr getGemvFunction() {
#if defined(PRECISION_FLOAT)
    return cublasSgemv;
#elif defined(PRECISION_DOUBLE)
    return cublasDgemv;
#elif defined(PRECISION_FLOAT16)
    return [](cublasHandle_t handle, cublasOperation_t trans, int m, int n,
              const __half* alpha, const __half* A, int lda,
              const __half* x, int incx, const __half* beta,
              __half* y, int incy) -> cublasStatus_t {
        const __half* A_array[1] = {A};
        const __half* x_array[1] = {x};
        __half* y_array[1] = {y};
        return cublasHgemvBatched(handle, trans, m, n, alpha, A_array, lda,
                                  x_array, incx, beta, y_array, incy, 1);
    };#else
    return nullptr; // Should never reach here due to the preprocessor error
#endif
}

constexpr cublasGemvPtr cublasGemv = getGemvFunction();
constexpr cublasBatchedQRPtr cublasBatchedComplexQR = getBatchedQRFunction();
constexpr cublasNormPtr cublasNorm = getNormFunction();
constexpr cublasScalePtr cublasScale = getScaleFunction();


// ==================== LINALG ROUTINES ====================

inline void cublasMGS(cublasHandle_t handle,
                     const HostPrecision* d_evecs,
                     HostPrecision* d_h,
                     HostPrecision* d_result,
                     int N,
                     int num_iters,
                     int i) {
    constexpr HostPrecision NEG_ONE = -1.0;
    constexpr HostPrecision ONE = 1.0;
    constexpr HostPrecision ZERO = 0.0;
    for (int j = 0; j <= i; j++) {
        // Compute projection coefficient <v_j, w> and store in H(j,i)
        cublasGemv(handle, CUBLAS_OP_T, N, 1, &ONE,
                   &d_evecs[j * N], N, d_result, 1,
                   &ZERO, &d_h[i * (num_iters + 1) + j], 1);
        
        // Subtract projection: w = w - h_ij * v_j
        cublasGemv(handle, CUBLAS_OP_N, N, 1, &NEG_ONE,
                   &d_evecs[j * N], N, &d_h[i * (num_iters + 1) + j], 1,
                   &ONE, d_result, 1);
    }
}


#endif // CUDA_MANAGER_HPP