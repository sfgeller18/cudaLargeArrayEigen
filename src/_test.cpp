#include <iostream>
#include <Eigen/Dense>
#include <array>
#include <random>
#include "arnoldi.hpp"
#include "eigenSolver.hpp"
#include <chrono>

#include "tests.hpp"

bool areColumnsOrthogonal(const MatrixColMajor& matrix) {
    int cols = matrix.cols();
    MatrixColMajor innerProducts = MatrixColMajor::Zero(cols, cols); // Matrix to store inner products

    // Check pairs of columns for orthogonality
    for (int i = 0; i < cols; ++i) {
        for (int j = 0; j < cols; ++j) {
            // Calculate the dot product of column i and column j
            innerProducts(i, j) = matrix.col(i).dot(matrix.col(j));
        }
    }

    // Print the matrix of inner products
    std::cout << "Matrix of Inner Products:\n" << innerProducts << std::endl;

    // Check orthogonality
    for (int i = 0; i < cols; ++i) {
        for (int j = i + 1; j < cols; ++j) {
            if (std::abs(innerProducts(i, j)) > 1e-3) { // Tolerance for floating-point comparison
                return false; // Columns are not orthogonal
            }
        }
    }
    return true; // All columns are orthogonal
}


int main(int argc, char* argv[]) {
    // Default values
    int rows = 5;
    int cols = 6;

    // Check if arguments were provided
    if (argc > 2) {
        rows = std::atoi(argv[1]); // Convert the first argument to an integer
        cols = std::atoi(argv[2]); // Convert the second argument to an integer
    }

    // Print the values for verification
    std::cout << "Running matrix multiplication tests with " << rows << " rows and " << cols << " columns.\n";

    // Run the tests with the specified dimensions
    run_matmul_tests(rows, cols);
    return 0;
}

// int main() {
//     // Matrix dimensions and Arnoldi iteration parameters
//     const size_t N = 5;
//     const size_t max_iters = 3;
//     const size_t top_eigenpairs = 5;
//     const HostPrecision tol = 0.1;

//     // Create a random 1000x1000 matrix
//     Matrix M(N, N);
//     unsigned seed = static_cast<unsigned>(std::chrono::system_clock::now().time_since_epoch().count());
//     std::mt19937 gen(seed);  
//     std::uniform_real_distribution<HostPrecision> dis(0.0, 1.0); // Uniform distribution between 0 and 1

//     for (size_t i = 0; i < N; ++i) {
//         for (size_t j = 0; j < N; ++j) {
//             M(i, j) = dis(gen); // Fill the matrix with random values
//         }
//     }

//       // Assuming Matrix has a data() method returning float*

//     // Run Arnoldi iteration
//     auto eigenpairs = arnoldiEigen(M, max_iters, tol);

//     std::cout << areColumnsOrthogonal(eigenpairs.Q);

//     print(eigenpairs.Q);
//     print(eigenpairs.H_tilde);

//     return 0;
// }



// int main() {
//     int N = 1000; // Size of the original matrix
//     int m = 10;   // Number of Arnoldi iterations

//     // Allocate and initialize Q and H_tilde on the host
//     float* Q = new float[N * (m + 1)];
//     float* H_tilde = new float[m * (m + 1)];

//     // Initialize Q and H_tilde (you should fill these with your actual data)
//     // For this example, we'll just use random data
//     for (int i = 0; i < N * (m + 1); ++i) Q[i] = static_cast<float>(rand()) / RAND_MAX;
//     for (int i = 0; i < m * (m + 1); ++i) H_tilde[i] = static_cast<float>(rand()) / RAND_MAX;

//     // Compute Ritz pairs
//     int result = computeRitzPairs(Q, H_tilde, N, m);

//     // Clean up
//     delete[] Q;
//     delete[] H_tilde;

//     return result;
// }

MatrixColMajor generateRandomHessenbergMatrix(size_t n) {
    MatrixColMajor H = MatrixColMajor::Random(n, n);  // Random matrix

    // Set elements below the first sub-diagonal to zero to make it Hessenberg
    for (size_t i = 2; i < n; ++i) {
        for (size_t j = 0; j < i - 1; ++j) {
            H(i, j) = 0.0;  // Zero out elements below the first sub-diagonal
        }
    }

    return H;
}

// int main() {
//     // Generate a random 10x10 Hessenberg matrix
//     size_t n = 10;
//     MatrixColMajor H = generateRandomHessenbergMatrix(n);

//     // Print the generated Hessenberg matrix

//     auto [eigenvalues, eigenvectors] = HessenbergEigenvaluesAndVectors(H, n);

//     // Print the sorted eigenvalues
//     std::cout << "Sorted Eigenvalues:\n";
//     print(eigenvalues);

//     // Print the sorted eigenvectors
//     std::cout << "Sorted Eigenvectors:\n";
//     print(complexToRealMatrix(eigenvectors));


//     return 0;
// }