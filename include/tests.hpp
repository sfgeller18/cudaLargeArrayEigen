#ifndef TESTS_HPP
#define TESTS_HPP

    #include <iostream>
    #include <cstring>
    #include "vector.hpp"
    #include "eigenSolver.hpp"
    #include "arnoldi.hpp"
    #include "matmul.hpp"
    #include <chrono>
    #include <cmath>


    template <typename MatType>
    MatType generateRandomHessenbergMatrix(size_t N) {
        MatType H = MatType::Random(N, N);
        // Zero out elements below the first subdiagonal
        for (size_t i = 2; i < N; ++i) {
            for (size_t j = 0; j < i - 1; ++j) {
                H(i, j) = 0.0;
            }
        }
        return H;
    }

    bool is_approx_equal(const Vector& a, const Vector& b, float epsilon = 1e-2) {
        if (a.size() != b.size()) return false;
        for (size_t i = 0; i < a.size(); ++i) {
            if (std::abs(a[i] - b[i]) > epsilon) {
                return false;
            }
        }
        return true;
    }

    
    // Helper function to check if a vector is zero
    bool isZeroVector(const Vector& vec, const double tol = 1e-10) {
        return vec.norm() < tol;
    }

    // Helper function to check if two vectors are parallel
    bool areVectorsParallel(const Vector& v1, const Vector& v2, const double tol = 1e-10) {
        if (isZeroVector(v1) || isZeroVector(v2)) return false;
        Vector normalized1 = v1.normalized();
        Vector normalized2 = v2.normalized();
        return std::abs(std::abs(normalized1.dot(normalized2)) - 1.0) < tol;
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
            break;
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