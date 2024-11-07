#include "arnoldi.hpp"
#include <gtest/gtest.h>

constexpr size_t N = 1000; // Test Matrix Size
constexpr HostPrecision maxResidual = 0.1;

using ArnoldiTestType = ComplexMatrix;

inline void ASSERT_LE_WITH_TOL(HostPrecision value, HostPrecision reference, double tol) {
    ASSERT_TRUE(value <= reference + tol) << "Value " << value << " is not less than or equal to " << reference << " within tolerance " << tol;
}

template <typename MatrixType>
void testRitzPairs(const MatrixType& M, size_t max_iters, cublasHandle_t& handle, HostPrecision tol = 1e-5) {
    // Compute Ritz pairs
    constexpr bool isRowMajor = MatrixType::IsRowMajor;
    EigenPairs ritzPairs = NaiveArnoldi<MatrixType>(M, max_iters, handle);
    const ComplexVector& ritzValues = ritzPairs.values;      // Ritz eigenvalues
    const ComplexMatrix& ritzVectors = ritzPairs.vectors; // Ritz eigenvectors

    // Calculate matrix norm once
    HostPrecision matrix_norm = M.norm();

    const size_t R = max_iters < 10 ? max_iters : 10;
    
    // Verify each Ritz pair
    for (size_t i = 0; i < R; ++i) {
        const ComplexType& eigenvalue = ritzValues[i];
        ComplexMatrix H_shifted = M - eigenvalue * Matrix::Identity(ritzVectors.rows(), ritzVectors.rows());
        
        // Get current eigenvector
        ComplexVector current_vector;
        if (isRowMajor) {
            current_vector = ritzVectors.row(i).transpose();
        } else {
            current_vector = ritzVectors.col(i);
        }
        
        // Calculate residual and its norm
        ComplexVector residual = H_shifted * current_vector;
        HostPrecision residual_norm = residual.norm();
        HostPrecision vector_norm = current_vector.norm();
        
        // Calculate relative residuals
        HostPrecision relative_residual = residual_norm / (matrix_norm * vector_norm);
        HostPrecision scaled_residual = residual_norm / matrix_norm;
        
        // Output the results
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

template <size_t MAX_ITERS>
int ArnoldiTest(int argc, char* argv[]) {
    size_t matrixSize = (argc > 1) ? std::stoi(argv[1]) : 100; // Use the input size, or default to 100
    size_t max_iters = (argc > 2) ? std::stoi(argv[2]) : 50;
    size_t basis_size = (argc > 3) ? std::stoi(argv[3]) : 10;;


    max_iters = max_iters > MAX_ITERS ? MAX_ITERS : max_iters;


    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    // Create a test matrix M of size matrixSize
    MatrixColMajor M = MatrixColMajor::Random(matrixSize, matrixSize);

    // Start timing
    auto start = std::chrono::high_resolution_clock::now();

    // Run the test function
    testRitzPairs<MatrixColMajor>(M, max_iters, handle);

    // End timing
    auto end = std::chrono::high_resolution_clock::now();

    // Calculate duration in milliseconds
    std::chrono::duration<double, std::milli> duration = end - start;

    std::cout << "Time taken: " << duration.count() << " ms" << std::endl;

    cublasDestroy(handle);

    return 0;
}

int main(int argc, char* argv[]) {
    return ArnoldiTest<100>(argc, argv);
}
