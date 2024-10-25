#ifndef EIGENSOLVER_HPP
#define EIGENSOLVER_HPP

#include <iostream>
#include <chrono>
#include "lapack_manager.hpp"  // LAPACK
#include <cusolverDn.h> // cuSolver
#include <cuda_runtime.h>
#include <Eigen/Dense>  // For generating matrices
#include "vector.hpp"

#define LAPACK_CHECK(func_call) \
    { \
        int info = func_call; \
        if (info != 0) { \
            std::cerr << #func_call " failed with error code: " << info << std::endl; \
            return {}; \
        } \
    }

// Timing helper
using Clock = std::chrono::high_resolution_clock;

template <typename MatType>
std::tuple<ComplexVector, MatType> lapack_hessenberg_eigSolver(const MatType& H, const size_t& n) {
    constexpr bool isRowMajor = MatType::IsRowMajor;
    constexpr size_t LAPACK_MAT_TYPE = (isRowMajor) ? LAPACK_ROW_MAJOR : LAPACK_COL_MAJOR;
    constexpr char job = 'S'; // Compute Schur & Eigenvalues
    constexpr char compz = 'I';
    constexpr char side = 'R'; // Right Eigenvectors
    constexpr char howmny = 'B'; // All eigenvectors
    
    MatType H_copy = H;
    MatType Z(n, n);
    Vector wr(n), wi(n);
    Vector work(3 * n);
    IntVector select(n); 
    select.setOnes(); // Compute all eigenvectors
    Vector VL(1); // Not used for right eigenvectors
    Vector VR(n * n);
    int ldvl = 1, ldvr = n;
    int mm = n, m;



    int lda = n;

    // First, compute Schur decomposition and eigenvalues
    LAPACK_CHECK(EigenSolver(LAPACK_MAT_TYPE, job, compz, n, 1, n, H_copy.data(), lda, wr.data(), wi.data(), Z.data(), lda));
    #ifdef PRECISION_FLOAT
        LAPACK_CHECK(LAPACKE_strevc(LAPACK_MAT_TYPE, side, howmny, select.data(),
                                    n, H_copy.data(), lda, nullptr, 1, Z.data(), ldvr, n,
                                    &m));
    #elif PRECISION_DOUBLE
        LAPACK_CHECK(LAPACKE_dtrevc(LAPACK_MAT_TYPE, side, howmny, select.data(),
                                    n, H_copy.data(), lda, nullptr, n, Z.data(), ldvr, n,
                                    &m));
    #endif

    ComplexVector evals(n);
    for (int i = 0; i < n; i++) {evals[i] = ComplexNumber(wr[i], wi[i]);}
    if (isRowMajor) {Z.transposeInPlace();}

    return std::make_tuple(evals, Z);
}

// LAPACK-based Schur decomposition (shseqr)
// Pass Upper Hessenberg Mat & Size
// Pass junk copy of Matrix
template <typename MatType> 
RealEigenPairs<MatType> eigenSolver(const MatType& A) {
    const size_t& N = A.cols();
    constexpr bool isRowMajor = MatType::IsRowMajor;

    #define LAPACK_EIGSOLVER
    //#define EIGEN_EIGSOLVER    

    #ifdef EIGEN_EIGSOLVER
    Eigen::EigenSolver<MatType> solver(A);
    ComplexVector eigenvals = solver.eigenvalues();
    // using ComplexMatType = std::conditional_t<isRowMajor,
    //                                           ComplexRowMajorMatrix,
    //                                           ComplexColMajorMatrix>;

    MatType eigenvecs = solver.eigenvectors().real();
    if (isRowMajor) {eigenvecs.transposeInPlace();}
    #endif
    #ifdef LAPACK_EIGSOLVER
    ComplexVector eigenvals(N);
    MatType eigenvecs(N, N);
    std::tie(eigenvals, eigenvecs) = lapack_hessenberg_eigSolver(A, N);
    #endif
    size_t num_non_zero = 0;
    // print(eigenvals);
    // print(eigenvecs);

    for (int i = 0; i < eigenvecs.cols(); ++i) {
        if (eigenvals[i].imag() != 0.0) {
            eigenvals[i] = 0.0;
            if (isRowMajor) {eigenvecs.row(i) = Vector::Zero(N);}
            else {eigenvecs.col(i) = Vector::Zero(N);}
        }
        else {
        // #define DEBUG
        #ifdef DEBUG
        ComplexVector evec(N);
        if (isRowMajor) {evec = eigenvecs.row(i);}
        else {eigenvecs.col(i);}
        ComplexVector result = A * evec - eigenvals(i) * evec;
        double norm = result.norm();
        std::cout << "Norm of column " << i << ": " << norm << std::endl;
        #endif
        num_non_zero++;
    }
    }



    Vector rEvals(N);
    rEvals = eigenvals.real();

    IntVector idx(N);
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(),
          [&rEvals](size_t i1, size_t i2) {
              return std::norm(rEvals[i1]) > std::norm(rEvals[i2]);
          });

    Vector sorted_evals(N);
    MatType sorted_evecs(N, N);
    size_t count = 0;
    for (size_t i = 0; i < N; ++i) {
        sorted_evals[i] = rEvals[idx[i]];
        if constexpr (MatType::IsRowMajor) {
            sorted_evecs.row(i) = eigenvecs.row(idx[i]);
        } else {
            sorted_evecs.col(i) = eigenvecs.col(idx[i]);

        }
    }

    std::cout << num_non_zero << std::endl;
    if (isRowMajor) {return {sorted_evals.head(num_non_zero), sorted_evecs.block(0, 0, num_non_zero, N), num_non_zero};}
    else {return {sorted_evals.head(num_non_zero), sorted_evecs.block(0, 0, N, num_non_zero), num_non_zero};}
}

#endif // EIGENSOLVER_HPP