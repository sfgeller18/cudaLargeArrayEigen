#ifndef EIGENSOLVER_HPP
#define EIGENSOLVER_HPP

#include <iostream>
#include <chrono>
#include "lapack_manager.hpp"  // LAPACK
#include <cusolverDn.h> // cuSolver
#include <cuda_runtime.h>
#include <Eigen/Dense>  // For generating matrices

#define LAPACK_CHECK(func_call) \
    { \
        int info = func_call; \
        if (info != 0) { \
            std::cerr << #func_call " failed with error code: " << info << std::endl; \
            return {}; \
        } \
    }

// Function to generate a random upper Hessenberg matrix
MatrixColMajor generateHessenbergMatrix(int n) {
    MatrixColMajor A(n, n); // Create an n x n matrix

    // Random number generation setup
    std::random_device rd; // Obtain a random number generator from hardware
    std::mt19937 gen(rd()); // Seed the generator
    std::uniform_real_distribution<> dis(-1.0, 1.0); // Define the range for random values

    // Fill the matrix with random values
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            A(i, j) = dis(gen); // Assign a random value in the range [-1, 1]
        }
    }
    for (int i = 2; i < n; ++i) {
        for (int j = 0; j < i - 1; ++j) {
            A(i, j) = 0.0;
        }
    }
    return A;
}

// Timing helper
using Clock = std::chrono::high_resolution_clock;

// LAPACK-based Schur decomposition (shseqr)
// Pass Upper Hessenberg Mat & Size
// Pass junk copy of Matrix
std::pair<ComplexVector, ComplexMatrix> HessenbergEigenvaluesAndVectors(MatrixColMajor& A, const size_t& n) {
    Vector H = Eigen::Map<Vector>(A.data(), A.size()); // LAPACK expects column-major data
    Vector wr(n), wi(n), Z(n * n);
    Vector work(3*n);
    
    constexpr char job = 'S'; // Compute Schur & Eigenvalues
    constexpr char compz = 'I';
    constexpr char side = 'R'; /// Right Evecs
    constexpr char howmny = 'B'; // All evecs

    int lda = n;

    LAPACK_CHECK(EigenSolver(LAPACK_COL_MAJOR, job, compz, n, 1, n, H.data(), lda, wr.data(), wi.data(), Z.data(), lda));
    
    
    IntVector select(n, 1);  // Compute all eigenvectors
    Vector VL(1);  // Not referenced for right eigenvectors
    Vector VR(n * n);
    int ldvl = 1, ldvr = n;
    int mm = n, m;

    #ifdef PRECISION_FLOAT
        LAPACK_CHECK(LAPACKE_strevc(LAPACK_COL_MAJOR, side, howmny, select.data(),
                                    n, H.data(), lda, nullptr, 1, Z.data(), ldvr, n,
                                    &m));
    #elif PRECISION_DOUBLE
        LAPACK_CHECK(LAPACKE_dtrevc(LAPACK_COL_MAJOR, side, howmny, select.data(),
                                    n, H.data(), lda, nullptr, 1, Z.data(), ldvr, n,
                                    &m))
    #endif


        // Construct complex eigenvalues and eigenvectors
    ComplexVector evals(n);
    ComplexMatrix evecs(n, n);

    for (size_t i = 0; i < n; ++i) {
        evals[i] = std::complex<HostPrecision>(wr[i], wi[i]);
        if (wi[i] == 0.0) {
            // Real eigenvalue
            for (size_t j = 0; j < n; ++j) {
                evecs(j, i) = Z[j * n + i];
            }
        } // Consciously not storing any complex conjugate pairs as we hope to keep only real vecs
    }


    // Sort eigenvalues and eigenvectors by magnitude of eigenvalues
    IntVector idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(),
          [&evals](size_t i1, size_t i2) {
              return std::norm(evals[i1]) > std::norm(evals[i2]);
          });

    ComplexVector sorted_evals(n);
    ComplexMatrix sorted_evecs(n, n);
    for (size_t i = 0; i < n; ++i) {
        sorted_evals[i] = evals[idx[i]];
        for (size_t j = 0; j < n; ++j) {
            sorted_evecs(j, i) = evecs(j, idx[i]);  // This line is changed
        }
    }

    return {sorted_evals, sorted_evecs};
}


#endif // EIGENSOLVER_HPP