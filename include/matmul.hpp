#ifndef MATMUL_HPP
#define MATMUL_HPP

    #include "vector.hpp"
    #include <stdexcept>
    #include "errormsg.hpp"
    #include <cmath>
    #include <iostream>
    #include <variant>

    #include "cuda_manager.hpp"


// L is row number, N is col number
template <typename MatrixType>
inline void matmul_internal(const MatrixType& M, DevicePrecision* d_M, const DevicePrecision* d_y, DevicePrecision* d_result, size_t NUM_ARRAYS, size_t L, size_t N, const DevicePrecision& alpha, const DevicePrecision& beta, cublasHandle_t& handle) {
    size_t idx = 0;
    constexpr bool isRowMajor = Matrix::IsRowMajor;
    std::cout << L << " " << N << std::endl;
    size_t& iterIndSize = (isRowMajor) ? L : N;
    size_t& axisArraySize = (isRowMajor) ? N : L;
    
    if (isRowMajor) {
        while (idx < L) {
        size_t selectedRows = std::min(NUM_ARRAYS, L - idx);
        // std::cout << "Block Transfer " << selectedRows << " rows, starting from index " << idx << std::endl;
        cudaMemcpyChecked(d_M, M.data() + idx * N, selectedRows * N * PRECISION_SIZE, cudaMemcpyKind::cudaMemcpyHostToDevice);
        CHECK_CUBLAS(cublasGemv(handle, CUBLAS_OP_T, N, selectedRows, &alpha, d_M, N, d_y, 1, &beta, d_result + idx, 1));
        idx += selectedRows;
        }
    } else {
        bool batched = (NUM_ARRAYS < N);
        DevicePrecision* tempResult = cudaMallocChecked<DevicePrecision>(N * PRECISION_SIZE);
        while (idx < N) {
            size_t selectedCols = std::min(NUM_ARRAYS, N - idx);
            // std::cout << "Block Transfer " << selectedCols << " rows, starting from index " << idx << std::endl;
            cudaMemcpyChecked(d_M, M.data() + idx * L, selectedCols * L * PRECISION_SIZE, cudaMemcpyKind::cudaMemcpyHostToDevice);

            CHECK_CUBLAS(cublasGemv(handle, CUBLAS_OP_N, L, selectedCols, &alpha, d_M, L, d_y + idx, 1, &beta, tempResult, 1));
            if (batched) {
            #ifdef PRECISION_FLOAT
                CHECK_CUBLAS(cublasSaxpy(handle, L, &alpha, tempResult, 1, d_result, 1));
                #elif PRECISION_DOUBLE
                CHECK_CUBLAS(cublasDaxpy(handle, L, &alpha, tempResult, 1, d_result, 1)); 
                #endif   
            }  else {d_y = d_result;}
            idx += selectedCols;
        }
    }

    }

using AmbigType = std::variant<Vector, DevicePrecision*>;
template <typename MatrixType>
AmbigType matmul(const MatrixType& M, const Vector& y, const CuRetType retType = CuRetType::HOST) {
    CHECK_DIMS(M, y);
    size_t free_mem = 0;
    size_t total_mem = 0;
    cudaError_t error = cudaMemGetInfo(&free_mem, &total_mem);
    size_t N = 0; // Number of columns in M
    size_t L = 0; // Number of rows in M
    std::tie(L, N) = shape(M);
    // CuBLAS GEMV variables
    const DevicePrecision alpha = 1.0;
    const DevicePrecision beta = 0.0;
    size_t idx = 0;
    const size_t& iterIndSize =  (MatrixType::IsRowMajor) ? N : L;
    size_t MAX_ALLOC = MAX_ROW_ALLOC(free_mem, iterIndSize);
    // std::cout << "Max Num Rows is " << MAX_ALLOC << std::endl;
    // Allocate device memory for M, y, and result
    DevicePrecision* d_M = cudaMallocChecked<DevicePrecision>(MAX_ALLOC * iterIndSize * PRECISION_SIZE);
    DevicePrecision* d_y = cudaMallocChecked<DevicePrecision>(N * PRECISION_SIZE); //Might eventually make sense to reduce this to do chunked mem cpy but not for now
    DevicePrecision* d_result = cudaMallocChecked<DevicePrecision>(L * PRECISION_SIZE);
    Vector h_result(L);
    
    cublasHandle_t handle;
    cublasCreate(&handle);


    cudaMemcpyChecked(d_y, y.data(), N * PRECISION_SIZE, cudaMemcpyKind::cudaMemcpyHostToDevice);
    matmul_internal(M, d_M, d_y, d_result, MAX_ALLOC, L, N, alpha, beta, handle);
    cudaMemcpyChecked(h_result.data(), d_result, L * PRECISION_SIZE, cudaMemcpyKind::cudaMemcpyDeviceToHost);
    
    // Free device memory
    cudaFree(d_M);
    cudaFree(d_y);
    cublasDestroy(handle);

    if (retType == CuRetType::HOST) {
        Vector h_result(L);
        cudaMemcpyChecked(h_result.data(), d_result, L * PRECISION_SIZE, cudaMemcpyKind::cudaMemcpyDeviceToHost);
        cudaFree(d_result);
        return h_result;
    } else {
        return d_result;
    }
}

inline Vector matmulHost(const Matrix& M, const Vector& y) {return std::get<Vector>(matmul(M, y, CuRetType::HOST));}
inline DevicePrecision* matmulDevice(const Matrix& M, const Vector& y) {return std::get<DevicePrecision*>(matmul(M, y, CuRetType::DEVICE));}









#endif //MATMUL_HPP

