#include "eigenSolver.hpp"
#include <gtest/gtest.h>

using TestMatType = ComplexMatrix;

TEST(EigenSolverTests, HessenbergLapackEigenDecomp) {
    constexpr size_t N = 10;
    EigenPairs result;
    auto H = generateRandomHessenbergMatrix<TestMatType>(N);

    int status = HessenbergLapackEigenDecomp<TestMatType>(H, result, N);
    ASSERT_EQ(status, 0) << "Hessenberg eigen decomposition failed";
    
    // Use testEigenpairs to validate eigenpairs
    bool isValid = testEigenpairs<TestMatType, EigenPairs>(H, result);
    ASSERT_TRUE(isValid) << "Eigenpairs validation failed for Hessenberg decomposition.";
}

// Additional tests for RealSymmetricEigenDecomp
TEST(EigenSolverTests, RealSymmetricEigenDecomp) {
    constexpr size_t N = 10;
    RealEigenPairs result;
    auto A = generateRandomSymmetricMatrix<TestMatType>(N);

    int status = RealSymmetricEigenDecomp<TestMatType>(A, result, N);
    ASSERT_EQ(status, 0) << "Real symmetric eigen decomposition failed";
    
    // Validate eigenpairs
    bool isValid = testEigenpairs<TestMatType, RealEigenPairs>(A, result);
    ASSERT_TRUE(isValid) << "Eigenpairs validation failed for real symmetric decomposition.";
}

// Additional tests for HermitianEigenDecomp
TEST(EigenSolverTests, HermitianEigenDecomp) {
    constexpr size_t N = 10;
    MixedEigenPairs result;
    auto A = generateRandomHermitianMatrix<TestMatType>(N);

    int status = HermitianEigenDecomp<TestMatType>(A, result, N);
    ASSERT_EQ(status, 0) << "Hermitian eigen decomposition failed";
    
    bool isValid = testEigenpairs<TestMatType, MixedEigenPairs>(A, result);
    ASSERT_TRUE(isValid) << "Eigenpairs validation failed for Hermitian decomposition.";
}


int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
