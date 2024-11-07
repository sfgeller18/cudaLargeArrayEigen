#ifndef MATMUL_HPP
#define MATMUL_HPP

#include "vector.hpp"
#include <stdexcept>
#include "errormsg.hpp"
#include <cmath>
#include <iostream>
#include <variant>
#include "cuda_manager.hpp"

// Uncomment for debugging
//#define DEBUG_MATMUL

#ifdef DEBUG_MATMUL
#include <iomanip>
void dbg_check(DevicePrecision* d_M, 
               DevicePrecision* d_y, 
               DevicePrecision* d_result, 
               size_t selectedSize, 
               size_t idx, 
               size_t N, 
               size_t L, 
               bool isRowMajor) {
    // Create host copies for debugging
    MatrixType h_M_copy(selectedSize, (isRowMajor ? N : L));
    Vector h_y_copy(selectedSize);
    Vector h_result_copy(selectedSize);
    
    // Copy current block of d_M back to host
    cudaMemcpy(h_M_copy.data(), d_M, selectedSize * (isRowMajor ? N : L) * PRECISION_SIZE, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_y_copy.data(), d_y + (isRowMajor ? 0 : idx), selectedSize * PRECISION_SIZE, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_result_copy.data(), d_result + (isRowMajor ? 0 : idx), selectedSize * PRECISION_SIZE, cudaMemcpyDeviceToHost);

    // Print the copied values
    std::cout << "Current Block of M (Rows " << idx << " to " << idx + selectedSize - 1 << "):\n";
    print(h_M_copy);
    std::cout << "Current Block of y (Idx " << idx << " to " << idx + selectedSize - 1 << "):\n";
    print(h_y_copy);
    std::cout << "Current Block of result (Idx " << idx << " to " << idx + selectedSize - 1 << "):\n";
    print(h_result_copy);       
    std::cout << "\n";
}
#endif

using AmbigType = std::variant<Vector, DevicePrecision*>;

template <typename M>
struct MatMulTraits {
    using S = typename std::conditional_t<std::is_same_v<typename M::Scalar, ComplexType>, DeviceComplexType, DevicePrecision>;
};

template <typename M>
inline void matmul_internal(const M& Mat, typename MatMulTraits<M>::S* d_M, const typename MatMulTraits<M>::S* d_y, typename MatMulTraits<M>::S* d_result, size_t NUM_ARRAYS, size_t L, size_t N, cublasHandle_t& handle) {
    size_t idx = 0;
    using S = MatMulTraits<M>::S;
    constexpr bool isRowMajor = M::IsRowMajor;
    constexpr S alpha = getOne<S>();
    constexpr S beta = getZero<S>();
    // std::cout << L << " " << N << std::endl;
    size_t& iterIndSize = (isRowMajor) ? L : N;
    size_t& axisArraySize = (isRowMajor) ? N : L;
    

while (idx < (isRowMajor ? L : N)) {
    size_t selectedElements = std::min(NUM_ARRAYS, (isRowMajor ? L : N) - idx);
    cudaMemcpyChecked(d_M, Mat.data() + idx * (isRowMajor ? N : L), selectedElements * (isRowMajor ? N : L) * PRECISION_SIZE, cudaMemcpyKind::cudaMemcpyHostToDevice);
    cublas::gemv<S>(handle, isRowMajor ? CUBLAS_OP_T : CUBLAS_OP_N, isRowMajor ? N : L, selectedElements, &alpha, d_M, isRowMajor ? N : L, isRowMajor ? d_y : d_y + idx, 1, &beta, isRowMajor ? d_result + idx : d_result, 1);
    idx += selectedElements;
}

}

template <typename MatrixType>
AmbigType matmul(const MatrixType& M, const Vector& y, 
                 const CuRetType retType) {
    CHECK_DIMS(M, y);
    size_t N = 0, L = 0;
    std::tie(L, N) = shape(M);

    size_t iterIndSize = (MatrixType::IsRowMajor) ? N : L;
    size_t MAX_ALLOC = DYNAMIC_ROW_ALLOC(iterIndSize);

    DevicePrecision* d_M = cudaMallocChecked<DevicePrecision>(MAX_ALLOC * iterIndSize * PRECISION_SIZE);
    DevicePrecision* d_y = cudaMallocChecked<DevicePrecision>(N * PRECISION_SIZE);
    DevicePrecision* d_result = cudaMallocChecked<DevicePrecision>(L * PRECISION_SIZE);

    cublasHandle_t handle;
    cublasCreate(&handle);

    cudaMemcpyChecked(d_y, y.data(), N * PRECISION_SIZE, cudaMemcpyKind::cudaMemcpyHostToDevice);
    matmul_internal<MatrixType>(M, d_M, d_y, d_result, MAX_ALLOC, L, N, handle);
    
    cudaFree(d_M);
    cudaFree(d_y);
    cublasDestroy(handle);

    if (retType == CuRetType::HOST) {
        Vector h_result(L);
        cudaMemcpyChecked(h_result.data(), d_result, L * PRECISION_SIZE, cudaMemcpyDeviceToHost);
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


#endif // MATMUL_HPP



