#include "arnoldi.hpp"
#include "IRAM.hpp"
#include <gtest/gtest.h>

constexpr size_t N = 10000; // Test Matrix Size
constexpr size_t total_iters = 1000;
constexpr size_t max_iters = 50;
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
        double residual_norm = residual.norm();
        double vector_norm = ritzPairs.vectors.col(i).norm();
        double relative_residual = residual_norm / (matrix_norm * vector_norm);
        double scaled_residual = residual_norm / matrix_norm;

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
//     cublasCreate(&handle);
//     ArnoldiTestType M = ArnoldiTestType::Random(N, N);
//     ComplexEigenPairs ritzPairs = NaiveArnoldi<ArnoldiTestType, N, N, max_iters>(M, handle);
//     checkRitzPairs<ArnoldiTestType, ComplexEigenPairs>(M, ritzPairs);
//     cublasDestroy(handle);
// }

// TEST(ArnoldiTests, OrthonormalityTest) {
//     cublasHandle_t handle;
//     cublasCreate(&handle);
//     ArnoldiTestType M = ArnoldiTestType::Random(N, N);
//     KrylovPair<ArnoldiTestType::Scalar> arnoldiResult = KrylovIter<ArnoldiTestType, N, N, max_iters>(M, handle);
//     ASSERT_TRUE(isOrthonormal<ComplexMatrix>(arnoldiResult.Q)) << "The columns of Q are not orthonormal.";
//     cublasDestroy(handle);
// }

TEST(ArnoldiTests, IRAMTest) {
    cublasHandle_t handle;
    cusolverDnHandle_t solver_handle;
    cublasCreate(&handle);
    cusolverDnCreate(&solver_handle);
    ArnoldiTestType M = ArnoldiTestType::Random(N, N);
    HostPrecision matnorm = M.norm();
    for (auto& x : M.reshaped()) {x /= matnorm;}
    ComplexEigenPairs ritzPairs = IRAM<ArnoldiTestType, N, total_iters, max_iters, basis_size>(M, handle, solver_handle);
    checkRitzPairs<ArnoldiTestType, ComplexEigenPairs, true>(M, ritzPairs);
    cublasDestroy(handle);
    cusolverDnDestroy(solver_handle);
}



int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
