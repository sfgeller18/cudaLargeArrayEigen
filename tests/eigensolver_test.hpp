#ifndef EIGENSOLVER_TESTS_HPP
#define EIGENSOLVER_TESTS_HPP

    #include "eigensSolver.hpp"
    #include <gtest/gtest.h>

    using ComplexTestType = ComplexMatrix;
    using RealTestType = Matrix;

    constexpr size_t N = 500; //Test Matrix Size

    template <typename MatrixType, typename EigenPairType>
    bool testEigenpairs(const MatrixType& A, const EigenPairType& eigenPairs) {
        using ScalarType = typename MatrixType::Scalar;
        using SecondMatrixType = std::conditional_t<
            std::is_same<ScalarType, ComplexType>::value || 
            std::is_same<EigenPairType, EigenPairs>::value, 
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
        EigenPairs result;
        auto H = generateRandomHessenbergMatrix<ComplexTestType>(N);

        int status = HessenbergLapackEigenDecomp<ComplexTestType>(H, result, N);
        ASSERT_EQ(status, 0) << "Hessenberg eigen decomposition failed";
        
        // Use testEigenpairs to validate eigenpairs
        bool isValid = testEigenpairs<ComplexTestType, EigenPairs>(H, result);
        ASSERT_TRUE(isValid) << "Eigenpairs validation failed for Hessenberg decomposition.";
    }

    // Additional tests for RealSymmetricEigenDecomp
    TEST(EigenSolverTests, RealSymmetricEigenDecomp) {
        RealEigenPairs result;
        auto A = generateRandomSymmetricMatrix<RealTestType>(N);

        int status = RealSymmetricEigenDecomp<RealTestType>(A, result, N);
        ASSERT_EQ(status, 0) << "Real symmetric eigen decomposition failed";
        
        // Validate eigenpairs
        bool isValid = testEigenpairs<RealTestType, RealEigenPairs>(A, result);
        ASSERT_TRUE(isValid) << "Eigenpairs validation failed for real symmetric decomposition.";
    }

    // Additional tests for HermitianEigenDecomp
    TEST(EigenSolverTests, HermitianEigenDecomp) {
        MixedEigenPairs result;
        auto A = generateRandomHermitianMatrix<ComplexTestType>(N);

        int status = HermitianEigenDecomp<ComplexTestType>(A, result, N);
        ASSERT_EQ(status, 0) << "Hermitian eigen decomposition failed";
        
        bool isValid = testEigenpairs<ComplexTestType, MixedEigenPairs>(A, result);
        ASSERT_TRUE(isValid) << "Eigenpairs validation failed for Hermitian decomposition.";
    }


#endif // EIGENSOLVER_TESTS_HPP
