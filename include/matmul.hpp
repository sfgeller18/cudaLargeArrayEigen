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
inline void matmul_internal(const MatrixType& M, DevicePrecision* d_M, const DevicePrecision* d_y, DevicePrecision* d_result, size_t NUM_ARRAYS, size_t L, size_t N, cublasHandle_t& handle) {
    size_t idx = 0;
    constexpr bool isRowMajor = MatrixType::IsRowMajor;
    // std::cout << L << " " << N << std::endl;
    size_t& iterIndSize = (isRowMajor) ? L : N;
    size_t& axisArraySize = (isRowMajor) ? N : L;
    
    if (isRowMajor) {
        constexpr DevicePrecision alpha = 1.0;
        constexpr DevicePrecision beta = 0.0;
        while (idx < L) {
        size_t selectedRows = std::min(NUM_ARRAYS, L - idx);
        // std::cout << "Block Transfer " << selectedRows << " rows, starting from index " << idx << std::endl;
        cudaMemcpyChecked(d_M, M.data() + idx * N, selectedRows * N * PRECISION_SIZE, cudaMemcpyKind::cudaMemcpyHostToDevice);
        CHECK_CUBLAS(cublasGemv(handle, CUBLAS_OP_T, N, selectedRows, &alpha, d_M, N, d_y, 1, &beta, d_result + idx, 1));
        idx += selectedRows;
        }
    } else {
        constexpr DevicePrecision alpha = 1.0;
        constexpr DevicePrecision beta = 1.0;
        while (idx < N) {
            size_t selectedCols = std::min(NUM_ARRAYS, N - idx);
            // std::cout << "Block Transfer " << selectedCols << " rows, starting from index " << idx << std::endl;
            cudaMemcpyChecked(d_M, M.data() + idx * L, selectedCols * L * PRECISION_SIZE, cudaMemcpyKind::cudaMemcpyHostToDevice);
            CHECK_CUBLAS(cublasGemv(handle, CUBLAS_OP_N, L, selectedCols, &alpha, d_M, L, d_y + idx, 1, &beta, d_result, 1));
            idx += selectedCols;
        }
    }

    }

using AmbigType = std::variant<Vector, DevicePrecision*>;
template <typename MatrixType>
AmbigType matmul(const MatrixType& M, const Vector& y, const CuRetType retType = CuRetType::HOST) {
    std::cout << MatrixType::IsRowMajor << std::endl;
    CHECK_DIMS(M, y);
    size_t free_mem = 0;
    size_t total_mem = 0;
    cudaError_t error = cudaMemGetInfo(&free_mem, &total_mem);
    size_t N = 0; // Number of columns in M
    size_t L = 0; // Number of rows in M
    std::tie(L, N) = shape(M);

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
    matmul_internal<MatrixType>(M, d_M, d_y, d_result, MAX_ALLOC, L, N, handle);
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

template <typename MatrixType>
inline Vector matmulHost(const MatrixType& M, const Vector& y) {return std::get<Vector>(matmul<MatrixType>(M, y, CuRetType::HOST));}
template <typename MatrixType>
inline DevicePrecision* matmulDevice(const MatrixType& M, const Vector& y) {return std::get<DevicePrecision*>(matmul<MatrixType>(M, y, CuRetType::DEVICE));}









#endif //MATMUL_HPP

