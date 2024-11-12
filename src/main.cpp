#include "eigenSolver.hpp"
#include <gtest/gtest.h>
#include <eigen3/Eigen/Dense>

using ComplexTestType = ComplexMatrix;
using RealTestType = Matrix;

constexpr size_t N = 500; // Test Matrix Size
// constexpr double default_tol = 1e-10; // Tolerance for eigenpair validation

template <typename MatrixType, typename EigenPairType>
bool testEigenpairs(const MatrixType& A, const EigenPairType& eigenPairs) {
    using ScalarType = typename MatrixType::Scalar;
    using SecondMatrixType = std::conditional_t<
        std::is_same<ScalarType, ComplexType>::value, 
        ComplexMatrix, 
        MatrixType>;

    bool result = true;
    for (int i = 0; i < eigenPairs.num_pairs; ++i) {
        SecondMatrixType Av = A * eigenPairs.vectors.col(i);
        SecondMatrixType lambda_v = eigenPairs.values[i] * eigenPairs.vectors.col(i);
        HostPrecision error = (Av - lambda_v).norm();
        if (error > default_tol) {
            result = false;
            std::cout << "Failure at Eigenpair " << i + 1 << ". Error of: " << error << "\n";
            break;
        }
    }
    return result;
}

TEST(EigenSolverTests, HessenbergLapackEigenDecomp) {
    ComplexEigenPairs result;
    ComplexMatrix H = generateRandomHessenbergMatrix<ComplexMatrix>(N);
    hessEigSolver<ComplexMatrix>(H, result, N);
    
    bool isValid = testEigenpairs(H, result);
    ASSERT_TRUE(isValid) << "Eigenpairs validation failed for Hessenberg decomposition.";
}

TEST(EigenSolverTests, RealSymmetricEigenDecomp) {
    RealEigenPairs result;
    Matrix A = generateRandomSymmetricMatrix<Matrix>(N);
    realSymmetricEigSolver<Matrix>(A, result, N);    
    bool isValid = testEigenpairs(A, result);
    ASSERT_TRUE(isValid) << "Eigenpairs validation failed for real symmetric decomposition.";
}

TEST(EigenSolverTests, HermitianEigenDecomp) {
    MixedEigenPairs result;
    auto A = generateRandomHermitianMatrix<ComplexTestType>(N);
    hermitianEigSolver<ComplexMatrix>(A, result, N);
    bool isValid = testEigenpairs(A, result);
    ASSERT_TRUE(isValid) << "Eigenpairs validation failed for Hermitian decomposition.";
}


int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
