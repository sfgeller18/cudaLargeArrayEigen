#ifndef EIGENSOLVER_HPP
#define EIGENSOLVER_HPP

    #define LAPACK_EIGSOLVER
    //#define EIGEN_EIGSOLVER    
    //#define CUDA_EIGSOLVER

#include <lapack.hh>   // LAPACK++ 
#include "vector.hpp"
#include <complex>
#include <numeric>
#include <memory>
#include "utils.hpp"



enum matrix_type : char {
    HESSENBERG = 'H',
    SELFADJOINT = 'S',
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


// Potential In-Place Method for large arrays, will fix later
// inline void sortEigenPairs(EigenPairs& pair) {
//     ComplexVector& evals = pair.values;
//     ComplexMatrix& evecs = pair.vectors;
//     const size_t& N = pair.num_pairs;
//     std::vector<size_t> indices(N);
//     std::iota(indices.begin(), indices.end(), 0);

//     // Sort indices based on eigenvalues
//     std::sort(indices.begin(), indices.end(),
//               [&evals](size_t i1, size_t i2) {
//                   return std::norm(evals[i1]) > std::norm(evals[i2]);
//               });


//     std::vector<bool> visited(N, false);

//     for (size_t i = 0; i < N; ++i) {
//         if (visited[i] || indices[i] == i) {continue;}
//         size_t current = i;
//         while (!visited[current]) {
//             visited[current] = true;
//             size_t nextIndex = indices[current];

//             std::swap(evals[current], evals[nextIndex]);
//             evecs.col(current).swap(evecs.col(nextIndex));

//             current = nextIndex;
//         }
//     }
// }

inline void sortEigenPairs(EigenPairs& pair) {
    ComplexVector& evals = pair.values;
    ComplexMatrix& evecs = pair.vectors;
    const size_t N = pair.num_pairs;

    // Create a vector of indices
    std::vector<size_t> indices(N);
    std::iota(indices.begin(), indices.end(), 0);

    // Sort indices based on eigenvalues
    std::sort(indices.begin(), indices.end(),
              [&evals](size_t i1, size_t i2) {
                  return std::norm(evals[i1]) > std::norm(evals[i2]); // Sorting in descending order of norms
              });

    // Create sorted eigenvalues and eigenvectors based on sorted indices
    #ifdef INPLACE_SORT
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
    #else
    ComplexVector sorted_evals(N);
    ComplexMatrix sorted_evecs(evecs.rows(), evecs.cols());

    for (size_t i = 0; i < N; ++i) {
        sorted_evals[i] = evals[indices[i]];
        sorted_evecs.col(i) = evecs.col(indices[i]);
    }

    // Update the original pairs
    evals = sorted_evals;
    evecs = sorted_evecs;
    #endif
}


template <typename MatrixType>
inline int HessenbergLapackEigenDecomp(const MatrixType& eigenMatrix, EigenPairs& resultHolder, const size_t& n) {
    Eigen::MatrixXcd H = eigenMatrix;
    Eigen::VectorXcd w(n);
    Eigen::MatrixXcd Z(n, n);

    // TREVC Variables
    bool select[n]; // Use a plain array
    std::fill(select, select + n, true);

    Eigen::MatrixXcd VR(n, n);
    int64_t m = 0;

    // Call LAPACK functions
    LAPACKPP_CHECK(lapack::hseqr(lapack::JobSchur::Schur, lapack::Job::Vec, n, 1, n, H.data(), n, w.data(), Z.data(), n));
    LAPACKPP_CHECK(lapack::trevc3(lapack::Sides::Right, lapack::HowMany::All, select, n, H.data(), n, nullptr, n, VR.data(), n, n, &m));

    ComplexMatrix evecs = Z * VR; // Ensure ComplexMatrix is defined correctly

    resultHolder = {w, evecs, false, false, n};

    return 0; // Return success
}

// #ifdef EIGEN_EIGSOLVER
template <typename MatType>
inline int complex_eigen_eigsolver(const MatType& A, EigenPairs& resultHolder, const size_t& N) {
using EigenSolver = std::conditional_t<std::is_same_v<typename MatType::Scalar, ComplexType>, 
                                        Eigen::ComplexEigenSolver<MatType>, 
                                        Eigen::EigenSolver<MatType>>;
    EigenSolver solver(A);
    resultHolder = {solver.eigenvalues(), solver.eigenvectors(),false, false, N};
    return 0;
}

template <typename MatType>
inline int RealSymmetricEigenDecomp(const MatType& A, RealEigenPairs& resultHolder, const size_t& N) {
    static_assert(std::is_same<typename MatType::Scalar, ComplexType>::value == false, 
                  "Only use RealSymmetricEigenDecomp for Real MatrixType");
    constexpr bool isRowMajor = MatType::IsRowMajor;
    MatType H = A;
    Vector w(N);
    LAPACKPP_CHECK(lapack::syev(lapack::Job::Vec, lapack::Uplo::Upper, N, H.data(), N, w.data()));
    resultHolder = {w, H, N};
    return 0;
}

template <typename MatType>
inline int HermitianEigenDecomp(const MatType& A, MixedEigenPairs& resultHolder, const size_t& N) {
    static_assert(std::is_same<typename MatType::Scalar, ComplexType>::value == true, 
                  "Only use HermitianEigenDecomp for Complex MatrixType");
    constexpr bool isRowMajor = MatType::IsRowMajor;
    MatType H = A;
    Vector w(N);
    LAPACKPP_CHECK(lapack::heev(lapack::Job::Vec, lapack::Uplo::Upper, N, H.data(), N, w.data()));
    resultHolder = {w, H, N};
    return 0;
}






#include <iostream>



template <typename MatType>
inline void eigsolver(const MatType& A, EigenPairs& resultHolder, const size_t& N, 
                      const matrix_type type = matrix_type::REGULAR) {
    if (type == matrix_type::HESSENBERG) {
        HessenbergLapackEigenDecomp<MatType>(A, resultHolder, N);
    } else if (type == matrix_type::SELFADJOINT) {
        if constexpr (std::is_same_v<typename MatType::Scalar, ComplexType>) {
            MixedEigenPairs tempHolder{};
            HermitianEigenDecomp<MatType>(A, tempHolder, N);
            resultHolder.values = tempHolder.values;
            resultHolder.vectors = tempHolder.vectors;
            resultHolder.realEvals = true;
            resultHolder.realEvecs = true;
            resultHolder.num_pairs = N;

        } else {
            RealEigenPairs tempHolder{};
           RealSymmetricEigenDecomp<MatType>(A, tempHolder, N);
            resultHolder.values = tempHolder.values;
            resultHolder.vectors = tempHolder.vectors;
            resultHolder.realEvals = true;
            resultHolder.realEvecs = false;
            resultHolder.num_pairs = N;
        }
    } else {
        complex_eigen_eigsolver<MatType>(A, resultHolder, N);
    }
    sortEigenPairs(resultHolder);
}





#endif // EIGENSOLVER_HPP