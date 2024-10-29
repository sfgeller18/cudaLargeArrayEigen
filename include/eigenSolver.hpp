#ifndef EIGENSOLVER_HPP
#define EIGENSOLVER_HPP

    #define LAPACK_EIGSOLVER
    //#define EIGEN_EIGSOLVER    
    //#define CUDA_EIGSOLVER

#include <iostream>
#include <chrono>
#include <lapack.hh>   // LAPACK++ 
#include <cusolverDn.h> // cuSolver
#include <cuda_runtime.h>
#include <Eigen/Dense>  // For generating matrices
#include "vector.hpp"

// Timing helper
using Clock = std::chrono::high_resolution_clock;


enum matrix_type : char {
    HOUSEHOLDER = 'H',
    HERMITIAN = 'S',
    REGULAR = 'R'
};

RealEigenPairs purgeComplex(const EigenPairs& pair, const double& tol = 1e-10) {
    const ComplexVector& eigenvals = pair.values;
    const ComplexMatrix& eigenvecs = pair.vectors;
    const size_t& N = pair.num_pairs;
    
    Vector real_evals(N);
    Matrix real_evecs(N, N);
    size_t count = 0;

    for (size_t i = 0; i < N; ++i) {
        if (std::abs(eigenvals[i].imag()) < tol) {
            real_evals[count] = eigenvals[i].real();
            real_evecs.col(count) = eigenvecs.col(i).real();
            ++count;
        }
    }

    real_evals.conservativeResize(count);
    real_evecs.conservativeResize(N, count);

    return {real_evals, real_evecs, count};
}

inline void sortEigenPairs(EigenPairs& pair) {
    ComplexVector& evals = pair.values;
    ComplexMatrix& evecs = pair.vectors;
    const size_t& N = pair.num_pairs;
    std::vector<size_t> indices(N);
    std::iota(indices.begin(), indices.end(), 0);

    // Sort indices based on eigenvalues
    std::sort(indices.begin(), indices.end(),
              [&evals](size_t i1, size_t i2) {
                  return std::norm(evals[i1]) > std::norm(evals[i2]);
              });


    std::vector<bool> visited(N, false);

    for (size_t i = 0; i < N; ++i) {
        if (visited[i] || indices[i] == i) {continue;}
        size_t current = i;
        while (!visited[current]) {
            visited[current] = true;
            size_t nextIndex = indices[current];

            std::swap(evals[current], evals[nextIndex]);
            evecs.col(current).swap(evecs.col(nextIndex));

            current = nextIndex;
        }
    }
}


template <typename MatrixType>
int complexLapackEigenDecomp(const MatrixType& eigenMatrix, EigenPairs& resultHolder, const size_t& n) {
    Eigen::MatrixXcd H = eigenMatrix;
    Eigen::VectorXcd w(n);
    Eigen::MatrixXcd Z(n, n);

    //TREVC Variables
    bool* select = new bool[n];
    std::fill(select, select + n, true); // Select all eigenvalues
    Eigen::MatrixXcd VL(n, n);            // Left eigenvectors (not used)
    Eigen::MatrixXcd VR(n, n);            // Right eigenvectors
    int64_t m = 0;

    int64_t info = lapack::hseqr(lapack::JobSchur::Schur, lapack::Job::Vec, n, 1, n, H.data(), n, w.data(), Z.data(), n);
    int64_t info_trevc = lapack::trevc(lapack::Sides::Right, lapack::HowMany::All, select, n, H.data(), n, nullptr, n, VR.data(), n, n, &m);
    H = Z * VR;
    // blas::gemm(1.0, Z, VR, 0.0, H); // blaspp::gemm is too complicated for my little math brain

    delete[] select; // Remember to free the dynamically allocated memory
    resultHolder = {w, H, n};
    return 0;
}

// #ifdef EIGEN_EIGSOLVER
template <typename MatType>
inline void pure_eigen_eigsolver(const MatType& A, EigenPairs& resultHolder) {
    Eigen::EigenSolver<MatType> solver(A);
    resultHolder.values = solver.eigenvalues();
    resultHolder.vectors = solver.eigenvectors();
}



//Main interface for getting sorted eigenpairs
template <typename MatType>
inline void eigsolver(const MatType& A, EigenPairs& resultHolder, const size_t& N, const matrix_type& type = matrix_type::REGULAR) {
    if (type == matrix_type::HOUSEHOLDER) {
        complexLapackEigenDecomp<MatType>(A, resultHolder, N);
    } else if (type == matrix_type::HERMITIAN) {
        // Uncomment and implement realLapackEigenDecomp for hermitian case
        // realLapackEigenDecomp(A, resultHolder);
    } else {
        pure_eigen_eigsolver<MatType>(A, resultHolder);
    }
    sortEigenPairs(resultHolder);
}





#endif // EIGENSOLVER_HPP