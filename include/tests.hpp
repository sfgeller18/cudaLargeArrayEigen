#ifndef TESTS_HPP
#define TESTS_HPP

    #include <iostream>
    #include <cstring>
    #include "vector.hpp"
    #include "matmul.hpp"
    #include <chrono>
    #include <cmath>

    bool is_approx_equal(const Vector& a, const Vector& b, float epsilon = 1e-2) {
        if (a.size() != b.size()) return false;
        for (size_t i = 0; i < a.size(); ++i) {
            if (std::abs(a[i] - b[i]) > epsilon) {
                return false;
            }
        }
        return true;
    }

    // Function to run the GPU matmul and log average speed
    double test_gpu_matmul_speed(const Matrix& M, const Vector& y) {
        double total_time = 0.0;

        std::cout << "Running matrix multiplication 10 times on GPU..." << std::endl;
        for (int run = 0; run < 10; ++run) {
            auto start = std::chrono::high_resolution_clock::now();  // Start timing

            Vector result_gpu = matmulHost(M, y);  // GPU matmul

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


void run_matmul_tests(const size_t& rows, const size_t& cols) {
    // Define a small matrix and vector for testing
    
    // Initialize a matrix M (4x3)
    MatrixRowMajor M(rows, cols);
    std::memset(M.data(), 0, rows * cols * sizeof(HostPrecision));
    // std::fill(M.data(), M.data() + (rows * cols), static_cast<HostPrecision>(1.0));    std::cout << MatrixColMajor::IsRowMajor << std::endl;


    // Initialize a vector y (3)
    Vector y(cols);
    for (size_t i = 0; i < cols; ++i) {
        y[i] = static_cast<DevicePrecision>(i + 1); // Fill with some values
    }
    // Run the GPU matrix multiplication and log average speed
    double avg_time = test_gpu_matmul_speed(M, y);

    // Compute the CPU result for correctness check
    Vector gpu_result = std::get<Vector>(matmul(M, y)); // Ensure to call matmul again to get the result
    if (rows * cols < 1e8) {
    Vector cpu_result(rows); // Initialize a result vector (4)
    cpu_result = M * y;

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