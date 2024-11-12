#include "arnoldi.hpp"
#include <gtest/gtest.h>

constexpr size_t N = 1000; // Test Matrix Size
constexpr HostPrecision maxResidual = 0.1;

using ArnoldiTestType = ComplexMatrix;

inline void ASSERT_LE_WITH_TOL(HostPrecision value, HostPrecision reference, double tol) {
    ASSERT_TRUE(value <= reference + tol) << "Value " << value << " is not less than or equal to " << reference << " within tolerance " << tol;
}

// Checks for residuals <= max_residual
template <typename MatrixType>
void checkRitzPairs(const MatrixType& M, const ComplexEigenPairs& ritzPairs, const double tol = default_tol, const double max_residual = maxResidual) {
    double matrix_norm = M.norm();
    constexpr bool isComplex = std::is_same_v<typename MatrixType::Scalar, ComplexType>;

    // Capture ritzPairs to access values within the lambda
    auto nth_norm = [&ritzPairs, isComplex](size_t i) -> HostPrecision {
        return isComplex ? std::norm(ritzPairs.values[i]) : std::abs(ritzPairs.values[i]);
    };

    for (int i = 0; i < std::min<int>(10, ritzPairs.values.size()); ++i) {
        ComplexType eigenvalue = ritzPairs.values[i];
        ComplexVector residual = (M * ritzPairs.vectors.col(i)) - eigenvalue * ritzPairs.vectors.col(i);
        double relative_residual = residual.norm() / (matrix_norm * ritzPairs.vectors.col(i).norm());

        // Check if the relative residual is below the threshold
        ASSERT_LT(relative_residual, max_residual) << "Relative residual for Ritz pair " << i << " exceeds threshold.";

        // Check the ordering of norms
        if (i > 0) {ASSERT_LE_WITH_TOL(nth_norm(i), nth_norm(i - 1), default_tol);}
    }
}

TEST(ArnoldiTests, RitzPairsResidualTest) {
    ArnoldiTestType M = ArnoldiTestType::Random(N, N);
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    // Define parameters
    size_t max_iters = std::min(size_t(100), N - 1);
    size_t basis_size = 10;

    // Compute Ritz pairs
    auto ritzPairs = NaiveArnoldi<ArnoldiTestType>(M, max_iters, handle);
    CHECK_CUBLAS(cublasDestroy(handle));

    // Check residuals
    checkRitzPairs<ArnoldiTestType>(M, ritzPairs);
}

TEST(ArnoldiTests, OrthonormalityTest) {
    ArnoldiTestType M = ArnoldiTestType::Random(N, N);
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    KrylovPair<ArnoldiTestType::Scalar> arnoldiResult = KrylovIter<ArnoldiTestType>(M, 5, handle);
    CHECK_CUBLAS(cublasDestroy(handle));
    // Assert that Q is orthonormal within the specified tolerance
    ASSERT_TRUE(isOrthonormal<ComplexMatrix>(arnoldiResult.Q)) << "The columns of Q are not orthonormal.";
}




int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
