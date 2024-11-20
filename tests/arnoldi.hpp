#ifndef ARNOLDI_TEST_HPP
#define ARNOLDI_TEST_HPP

#include "arnoldi.hpp"
#include <gtest/gtest.h>
#include <eigen3/Eigen/Dense>  // Include necessary Eigen headers
#include <cublas_v2.h> // Include necessary CUDA headers

constexpr size_t N = 1000; // Test Matrix Size
constexpr size_t max_iters = 100;
constexpr size_t basis_size = 10;

constexpr HostPrecision maxResidual = 0.1;

using ArnoldiTestType = ComplexMatrix;

inline void ASSERT_LE_WITH_TOL(HostPrecision value, HostPrecision reference, double tol) {
    ASSERT_TRUE(value <= reference + tol) << "Value " << value << " is not less than or equal to " << reference << " within tolerance " << tol;
}

// Checks for residuals <= max_residual
template <typename MatrixType, typename PT, bool verbose=false>
void checkRitzPairs(const MatrixType& M, const PT& ritzPairs, const double tol = default_tol, const double max_residual = maxResidual) {
    double matrix_norm = M.norm();
    constexpr bool isComplex = std::is_same_v<typename MatrixType::Scalar, ComplexType>;

    auto nth_norm = [&ritzPairs, isComplex](size_t i) -> HostPrecision {
        return isComplex ? std::norm(ritzPairs.values[i]) : std::abs(ritzPairs.values[i]);
    };

    for (int i = 0; i < std::min<int>(10, ritzPairs.values.size()); ++i) {
        ComplexType eigenvalue = ritzPairs.values[i];
        ComplexVector residual = (M * ritzPairs.vectors.col(i)) - eigenvalue * ritzPairs.vectors.col(i);
        double relative_residual = residual.norm() / (matrix_norm * ritzPairs.vectors.col(i).norm());
        double scaled_residual = residual.norm() / matrix_norm;

        ASSERT_LT(relative_residual, max_residual) << "Relative residual for Ritz pair " << i << " exceeds threshold.";

        // Check the ordering of norms
        if (i > 0) {ASSERT_LE_WITH_TOL(nth_norm(i), nth_norm(i - 1), default_tol);}
        
        if (verbose) {
            std::cout << "Ritz pair " << i + 1 << ":" << std::endl;
            std::cout << "  Ritz value: " << eigenvalue << std::endl;
            std::cout << "  Absolute residual norm: " << residual_norm << std::endl;
            std::cout << "  Relative residual (||Ax - λx||/(||A|| ||x||)): " << relative_residual << std::endl;
            std::cout << "  Scaled residual (||Ax - λx||/||A||): " << scaled_residual << std::endl;
            std::cout << "  Matrix norm: " << matrix_norm << std::endl;
            std::cout << "  Vector norm: " << vector_norm << std::endl;
            std::cout << std::endl;
        }
    }
}


// TEST(ArnoldiTests, RitzPairsResidualTest) {
//     cublasHandle_t handle;
//     CHECK_CUBLAS(cublasCreate(&handle));
//     ArnoldiTestType M = ArnoldiTestType::Random(N, N);
//     ComplexEigenPairs ritzPairs = NaiveArnoldi<ArnoldiTestType, N, N, max_iters>(M, handle);
//     checkRitzPairs<ArnoldiTestType, ComplexEigenPairs>(M, ritzPairs);
//     CHECK_CUBLAS(cublasDestroy(handle));
// }

// TEST(ArnoldiTests, OrthonormalityTest) {
//     cublasHandle_t handle;
//     CHECK_CUBLAS(cublasCreate(&handle));
//     ArnoldiTestType M = ArnoldiTestType::Random(N, N);
//     KrylovPair<ArnoldiTestType::Scalar> arnoldiResult = KrylovIter<ArnoldiTestType, N, N, max_iters>(M, handle);
//     ASSERT_TRUE(isOrthonormal<ComplexMatrix>(arnoldiResult.Q)) << "The columns of Q are not orthonormal.";
//     CHECK_CUBLAS(cublasDestroy(handle));
// }

TEST(ArnoldiTests, IRAMTest) {
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));
    ArnoldiTestType M = ArnoldiTestType::Random(N, N);
    ComplexEigenPairs ritzPairs = IRAM<ArnoldiTestType, N, basis_size, max_iters>(M, handle);
    checkRitzPairs<ArnoldiTestType, ComplexEigenPairs, true>(M, ritzPairs);
    CHECK_CUBLAS(cublasDestroy(handle));
}




#endif //ARNOLDI_TEST_HPP