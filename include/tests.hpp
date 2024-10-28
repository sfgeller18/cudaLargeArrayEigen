#ifndef TESTS_HPP
#define TESTS_HPP

    #include "vector.hpp"
    #include "eigenSolver.hpp"
    #include "arnoldi.hpp"
    #include "matmul.hpp"
    #include "shift.hpp"


    #include "utils.hpp"

    
// ============================= SHIFTING TESTS =============================


// int shiftTests() {
//     const size_t& N = 10;
//     Matrix M = generateRandomHessenbergMatrix<Matrix>(N);
//     ComplexMatrix complex_M = M;
//     ComplexVector eigenvalues(N);
//     Matrix eigenvecs(N, N);
//     eigen_eigsolver(M, eigenvalues, eigenvecs);
//     ComplexMatrix S = ComplexMatrix::Identity(N,N);
//     computeShift(S, complex_M, eigenvalues, N, int(N/2));
//     print(S);
//     return 0;
// }

// ============================= RITZ PAIR TESTS =============================

template <typename MatrixType>
void testRitzPairs(const MatrixType& M, size_t max_iters, size_t basis_size, HostPrecision tol = 1e-5) {
    // Compute Ritz pairs
    constexpr bool isRowMajor = MatrixType::IsRowMajor;
    RealEigenPairs<MatrixType> ritzPairs = computeRitzPairs<MatrixType>(M, max_iters, basis_size, tol);
    const Vector& ritzValues = ritzPairs.values;      // Ritz eigenvalues
    const MatrixType& ritzVectors = ritzPairs.vectors; // Ritz eigenvectors

    // Calculate matrix norm once
    HostPrecision matrix_norm = M.norm();
    
    // Verify each Ritz pair
    for (size_t i = 0; i < ritzValues.size(); ++i) {
        const HostPrecision& eigenvalue = ritzValues[i];
        Matrix H_shifted = M - eigenvalue * Matrix::Identity(ritzVectors.rows(), ritzVectors.rows());
        
        // Get current eigenvector
        Vector current_vector;
        if (isRowMajor) {
            current_vector = ritzVectors.row(i).transpose();
        } else {
            current_vector = ritzVectors.col(i);
        }
        
        // Calculate residual and its norm
        Vector residual = H_shifted * current_vector;
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
    size_t max_iters = (matrixSize < 10 * MAX_ITERS) ? matrixSize / 10 : MAX_ITERS;
    size_t basis_size = 10;

    // Create a test matrix M of size matrixSize
    MatrixColMajor M = MatrixColMajor::Random(matrixSize, matrixSize);

    // Start timing
    auto start = std::chrono::high_resolution_clock::now();

    // Run the test function
    testRitzPairs(M, max_iters, basis_size);

    // End timing
    auto end = std::chrono::high_resolution_clock::now();

    // Calculate duration in milliseconds
    std::chrono::duration<double, std::milli> duration = end - start;

    std::cout << "Time taken: " << duration.count() << " ms" << std::endl;

    return 0;
}


// ============================= KRYLOV ITERATION TESTS =============================


    int iterationTest(int argc, char** argv) {
        // Check for the correct number of arguments
        if (argc < 2) {
            std::cerr << "Usage: " << argv[0] << " <matrix_size>" << std::endl;
            return EXIT_FAILURE;
        }

        // Parse the matrix size from command line arguments
        size_t N = std::atoi(argv[1]);

        // Generate a random matrix of size N x N
        MatrixColMajor M = MatrixColMajor::Random(N, N);

        // Set parameters for the Ritz pair computation
        size_t max_iters = 100; // Example maximum iterations
        size_t basis_size = 10; // Example basis size
        double tol = 1e-5;      // Tolerance for convergence

        // // Call the computeRitzPairs function
        // RealEigenPairs<MatrixColMajor> ritzPairs = computeRitzPairs(M, max_iters, basis_size, tol);

        // // Output the results
        // std::cout << "Eigenvalues:\n" << ritzPairs.values << std::endl;
        // std::cout << "Ritz Vectors:\n" << ritzPairs.vectors << std::endl;
        // std::cout << "Number of Eigen Pairs: " << ritzPairs.num_pairs << std::endl;

        // Check orthonormality of Q in the ArnoldiPair from arnoldiEigen
        KrylovPair arnoldiResult = krylovIter(M, std::min(max_iters, N), tol);
        if (isOrthonormal<MatrixColMajor>(arnoldiResult.Q)) {
            std::cout << "The columns of Q form an orthonormal set." << std::endl;
        } else {
            std::cout << "The columns of Q do not form an orthonormal set." << std::endl;
        }

        return EXIT_SUCCESS;
    }

// ============================= EIGENSOLVER TESTS =============================

template <typename MatType>
inline void testEpairs(const MatType& H, const MatType& Z, const Vector& evals, const size_t& num_pairs) {
    constexpr bool isRowMajor = MatType::IsRowMajor;
    bool correct = true;
    for (int i = 0; i < num_pairs; i++) {
        // if (evals[i].imag() != 0.0) {continue;}
        double norm_diff = isRowMajor ? (H * Z.row(i).transpose() - evals[i] * Z.row(i).transpose()).norm() : (H * Z.col(i) - evals[i] * Z.col(i)).norm();
        if (norm_diff > 1e-5) {
            std::cout << "Norm of difference for eigenpair " << i << ": " << norm_diff << std::endl;
            correct = false;
        }
    }

    if (correct) {std::cout << "All Eigenpairs Correct";}
}

    template <typename MatrixType>
    int eigSolverTest(int argc, char* argv[]) {
        // Check if the size argument is provided
        if (argc < 2) {
            std::cerr << "Usage: " << argv[0] << " <matrix_size>\n";
            return 1;
        }

        // Convert the command line argument to size_t
        size_t n = static_cast<size_t>(std::atoi(argv[1])); // Convert input argument to size_t

        if (n <= 0) {
            std::cerr << "Matrix size must be a positive integer.\n";
            return 1;
        }

        // Generate a random Hessenberg matrix of size n
        MatrixType H = generateRandomHessenbergMatrix<MatrixType>(n);

        // Perform eigenvalue decomposition
        RealEigenPairs<MatrixType> result = eigenSolver(H);
        const Vector& evals = result.values;
        const MatrixType& Z = result.vectors; 
        const size_t& num_pairs = result.num_pairs;

        // Test the eigenvalue pairs
        testEpairs(H, Z, evals, num_pairs);

        return 0;
    }

// ============================= MATMUL TESTS =============================

    // Function to run the GPU matmul and log average speed
    double test_gpu_matmul_speed(const Matrix& M, const Vector& y) {
        double total_time = 0.0;

        std::cout << "Running matrix multiplication 10 times on GPU..." << std::endl;
        for (int run = 0; run < 10; ++run) {
            auto start = std::chrono::high_resolution_clock::now();  // Start timing

            Vector result_gpu = matmulHost<Matrix>(M, y);  // GPU matmul

            auto end = std::chrono::high_resolution_clock::now();  // End timing
            std::chrono::duration<double> elapsed = end - start;
            total_time += elapsed.count();
        }

        double avg_time = total_time / 10.0;
        std::cout << "Average time for matrix multiplication over 10 runs: " << avg_time << " seconds" << std::endl;
        return avg_time;
    }

    // Function to check correctness of results
    inline bool check_correctness(const Vector& result_cpu, const Vector& result_gpu) {
        if (!is_approx_equal(result_cpu, result_gpu)) {
            std::cerr << "Test failed! CPU and GPU results differ." << std::endl;
            return false;
        } else {
            std::cout << "CPU and GPU results are approximately equal." << std::endl;
            return true;
        }
    }


    template <typename MatrixType>
    void run_matmul_tests(const size_t& rows, const size_t& cols) {
        // Define a small matrix and vector for testing
        
        // Initialize a matrix M (4x3)
        Matrix M(rows, cols);
        // std::memset(M.data(), 0, rows * cols * sizeof(HostPrecision));
        std::fill(M.data(), M.data() + (rows * cols), static_cast<HostPrecision>(1.0));
        std::cout << MatrixType::IsRowMajor << std::endl;


        // Initialize a vector y (3)
        Vector y(cols);
        for (size_t i = 0; i < cols; ++i) {
            y[i] = static_cast<DevicePrecision>(i + 1); // Fill with some values
        }
        // Run the GPU matrix multiplication and log average speed
        double avg_time = test_gpu_matmul_speed(M, y);

        // Compute the CPU result for correctness check
        Vector gpu_result = matmulHost<Matrix>(M, y); // Ensure to call matmul again to get the result
        if (rows * cols < 1e8) {
        Vector cpu_result(rows); // Initialize a result vector (4)
        cpu_result = M * y;
        print(cpu_result);
        print(M);
        std::cout << "M * y" << std::endl;

        // Get the GPU result

        if (rows * cols < 25) {
        print(cpu_result);
        print(gpu_result);
        }
        // Check correctness of GPU results against CPU results
        check_correctness(cpu_result, gpu_result);
    }
    }




#endif // TESTS_HPP