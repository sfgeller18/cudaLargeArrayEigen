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

// #define DEBUG_MATMUL

#ifdef DEBUG_MATMUL
#include <iomanip>
template <typename DS>
void dbg_check(const DS* d_M, 
               const DS* d_y, 
               const DS* d_result, 
               size_t selectedSize, 
               size_t idx, 
               size_t N, 
               size_t L, 
               bool isRowMajor) {
    // Create host copies for debugging
    using M = std::conditional_t<std::is_same_v<DS, DevicePrecision>, Matrix, ComplexMatrix>;
    using V = std::conditional_t<std::is_same_v<DS, DevicePrecision>, Vector, ComplexVector>;
    constexpr size_t ALLOC_SIZE = sizeof(DS);
    M h_M_copy(selectedSize, (isRowMajor ? N : L));
    V h_y_copy(selectedSize);
    V h_result_copy(selectedSize);
    
    // Copy current block of d_M back to host
    cudaMemcpy(h_M_copy.data(), d_M, selectedSize * (isRowMajor ? N : L) * ALLOC_SIZE, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_y_copy.data(), d_y + (isRowMajor ? 0 : idx), selectedSize * ALLOC_SIZE, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_result_copy.data(), d_result + (isRowMajor ? 0 : idx), selectedSize * ALLOC_SIZE, cudaMemcpyDeviceToHost);

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
template <typename V>
struct AmbigType;

template <>
struct AmbigType<Vector> {
    using Type = std::variant<Vector, DevicePrecision*>;
};

template <>
struct AmbigType<ComplexVector> {
    using Type = std::variant<ComplexVector, DeviceComplexType*>;
};

// Convenience alias
template <typename V>
using AmbigType_t = typename AmbigType<V>::Type;
template <typename M, typename S>
inline void matmul_internal(const M& M_, S* d_M, const S* d_y, S* d_result, size_t NUM_ARRAYS, size_t L, size_t N, cublasHandle_t& handle) {
    static_assert(std::is_same_v<typename M::Scalar, S> || 
              (std::is_same_v<typename M::Scalar, ComplexType> && std::is_same_v<S, DeviceComplexType>), 
              "Matrix and Vector types must match.");
    size_t idx = 0;
    constexpr bool isRowMajor = M::IsRowMajor;
    constexpr S ONE = getOne<S>();
    constexpr S ZERO = getZero<S>();
    constexpr size_t ALLOC_SIZE = sizeof(S);
    // std::cout << L << " " << N << std::endl;
    size_t& iterIndSize = (isRowMajor) ? L : N;
    size_t& axisArraySize = (isRowMajor) ? N : L;
    

while (idx < (isRowMajor ? L : N)) {
    size_t selectedElements = std::min(NUM_ARRAYS, (isRowMajor ? L : N) - idx);
    cudaMemcpyChecked(d_M, M_.data() + idx * (isRowMajor ? N : L), selectedElements * (isRowMajor ? N : L) * ALLOC_SIZE, cudaMemcpyKind::cudaMemcpyHostToDevice);
    CHECK_CUBLAS(cublas::gemv<S>(handle, isRowMajor ? CUBLAS_OP_T : CUBLAS_OP_N, isRowMajor ? N : L, selectedElements, &ONE, d_M, isRowMajor ? N : L, isRowMajor ? d_y : d_y + idx, 1, &ZERO, isRowMajor ? d_result + idx : d_result, 1));
    #ifdef DEBUG_MATMUL
    dbg_check<S>(d_M, d_y, d_result, selectedElements, idx, N, L, isRowMajor);
    #endif
    idx += selectedElements;
}

    }

template <typename M, typename V>
typename AmbigType<V>::Type matmul(const M& M_, const V& y, 
                 const CuRetType retType) {
    using S = typename M::Scalar;
    using DS = std::conditional_t<std::is_same_v<S, DevicePrecision>, DevicePrecision, DeviceComplexType>;
    constexpr size_t ALLOC_SIZE = sizeof(DS);
    static_assert(std::is_same<typename M::Scalar, typename V::Scalar>::value && "Matrix and Vector types must match."); //Temp until I add some casting logic
    // CHECK_DIMS(M_, y);
    size_t N = 0, L = 0;
    
    #ifdef USE_EIGEN
        N = M_.cols();
        L = M_.rows();
    #else
        N = cols(M_);
        L = rows(M_);
    #endif

    size_t iterIndSize = (M::IsRowMajor) ? N : L;
    size_t MAX_ALLOC = DYNAMIC_ROW_ALLOC(iterIndSize);

    DS* d_M = cudaMallocChecked<DS>(MAX_ALLOC * iterIndSize * ALLOC_SIZE);
    DS* d_y = cudaMallocChecked<DS>(N * ALLOC_SIZE);
    DS* d_result = cudaMallocChecked<DS>(L * ALLOC_SIZE);

    cublasHandle_t handle;
    cublasCreate(&handle);

    cudaMemcpyChecked(d_y, y.data(), N * ALLOC_SIZE, cudaMemcpyKind::cudaMemcpyHostToDevice);
    matmul_internal<M, DS>(M_, d_M, d_y, d_result, MAX_ALLOC, L, N, handle);
    
    cudaFree(d_M);
    cudaFree(d_y);
    cublasDestroy(handle);

    if (retType == CuRetType::HOST) {
        V h_result(L);
        cudaMemcpyChecked(h_result.data(), d_result, L * ALLOC_SIZE, cudaMemcpyDeviceToHost);
        cudaFree(d_result);
        return h_result;
    } else {
        return d_result;
    }
}

template <typename M, typename V>
inline V matmulHost(const M& M_, const V& y) {return std::get<V>(matmul<M, V>(M_, y, CuRetType::HOST));}

template <typename M, typename V>
inline auto matmulDevice(const M& M_, const V& y) {
    if constexpr (std::is_same_v<typename M::Scalar, DevicePrecision>) {
        return std::get<DevicePrecision*>(matmul<M, V>(M_, y, CuRetType::DEVICE));
    } else if constexpr (std::is_same_v<typename M::Scalar, DeviceComplexType>) {
        return std::get<DeviceComplexType*>(matmul<M, V>(M_, y, CuRetType::DEVICE));
    } else {
        static_assert(false, "Matrix type must have DevicePrecision or DeviceComplexType scalar");
    }
}
#endif // MATMUL_HPP



