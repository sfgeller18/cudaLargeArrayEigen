#include <iostream>
#include <vector>
#include <algorithm>
#include <cassert>
#include "shift.hpp"
#include "structs.hpp"
#include <chrono>

// #define CUBLAS_RESTART
#define EIGEN_RESTART

using MatType = Matrix;


//TO-DO: FIX WHAT Q MATRIX WE CONSERVATIVE RESIZE






int main(int argc, char* argv[]) {
    cublasHandle_t handle;
    cusolverDnHandle_t solver_handle;
    CHECK_CUBLAS(cublasCreate(&handle));
    CHECK_CUSOLVER(cusolverDnCreate(&solver_handle));

    
    const size_t& N = (argc > 1) ? std::stoi(argv[1]) : 10;
    const size_t& max_iters = (argc > 2) ? std::stoi(argv[2]) : 5;
    const size_t& basis_size = (argc > 3) ? std::stoi(argv[3]) : 3;
    
    MatType M;
    if (N > 10000) {
        // Allocate the matrix
        M = MatType::Zero(N, N); // Initialize with zeros first

        // Set all entries to -1
        typename MatType::Scalar* data = M.data();
        std::fill(data, data + M.size(), -1.0); // Using std::fill for safety and clarity
    } else {
        // If N is not greater than 10000, initialize normally
        M = MatType::Random(N, N);
    }
    std::cout << "M Initialized" << std::endl;
    const double tol = 1e-10 * M.norm();
    auto start = std::chrono::high_resolution_clock::now();
    ComplexKrylovPair q_h = KrylovIter<MatType>(M, max_iters, handle);
    print(q_h.H);
    print(q_h.Q);
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Time for Krylov Iter: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
              << " ms" << std::endl;
    std::cout << q_h.H << std::endl;
    Eigen::MatrixXcd S = Eigen::MatrixXcd::Identity(q_h.m, q_h.m);

    assert(isOrthonormal<ComplexMatrix>(q_h.Q));

    reduceArnoldiPair(q_h, basis_size, handle, solver_handle, resize_type::ZEROS);
   
    assert(isHessenberg<ComplexMatrix>(q_h.H));
    assert(isOrthonormal<ComplexMatrix>(q_h.Q.leftCols(basis_size), 1e-4));

    // // std::cout <<q_h.H << std::endl;

    // std::cout << "Arnoldi Reduction Test Passed!" << std::endl;
    CHECK_CUBLAS(cublasDestroy(handle));
    CHECK_CUSOLVER(cusolverDnDestroy(solver_handle));
    return 0;
}