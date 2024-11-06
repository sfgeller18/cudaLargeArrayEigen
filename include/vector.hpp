#ifndef VECTOR_HPP
#define VECTOR_HPP

#include <cstddef>
#include <iostream>
#include <utility>
#include <random>

#include <vector> // No matter what, necessary

#include <type_traits>
#include <complex.h>
#include <limits>



#if defined(PRECISION_FLOAT)
    using HostPrecision = float;
    using DevicePrecision = float;
#elif defined(PRECISION_DOUBLE)
    using HostPrecision = double;
    using DevicePrecision = double;
#elif defined(PRECISION_FLOAT16)
    #include <cuda_fp16.h>
    #include <stdfloat>
    using HostPrecision = std::float16_t; // or use float16_t if defined
    using DevicePrecision = __half;
#else
    #error "No precision defined! Please define PRECISION_FLOAT, PRECISION_DOUBLE, or PRECISION_FLOAT16."
#endif

using ComplexType = std::complex<HostPrecision>;
constexpr size_t PRECISION_SIZE = sizeof(HostPrecision);
constexpr HostPrecision default_tol = 1e-10;

// Conditional type definitions based on USE_EIGEN
#ifdef USE_EIGEN
    #include <eigen3/Eigen/Dense>

    // Vector (Column vector)
   using Vector = Eigen::Matrix<HostPrecision, Eigen::Dynamic, 1>; // Dynamic rows, single column
    using ComplexVector = Eigen::Matrix<std::complex<HostPrecision>, Eigen::Dynamic, 1>; // Complex column vector
    using IntVector = Eigen::Matrix<int, Eigen::Dynamic, 1>; // Integer column vector

    // Dynamic matrices
    using Matrix = Eigen::Matrix<HostPrecision, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>; // Default column-major matrix
    using ComplexMatrix = Eigen::Matrix<ComplexType, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>; // Complex matrix
    using MatrixRowMajor = Eigen::Matrix<HostPrecision, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>; // Row-major matrix
    using MatrixColMajor = Eigen::Matrix<HostPrecision, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>; // Column-major matrix
    using ComplexRowMajorMatrix = Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    using ComplexColMajorMatrix = ComplexMatrix;
    // Mapped types (Eigen::Map)
    using VectorMap = Eigen::Map<Vector>;
    using VectorMapConst = Eigen::Map<const Vector>;

    using MatrixMap = Eigen::Map<Matrix>;
    using MatrixColMajorMap = Eigen::Map<MatrixColMajor>;
    using MatrixMapConst = Eigen::Map<const Matrix>;

    
    using VectorMap = Eigen::Map<Vector>;
    using VectorMapConst = Eigen::Map<const Vector>;

    using MatrixMap = Eigen::Map<Matrix>;
    using MatrixColMajorMap = Eigen::Map<MatrixColMajor>;
    using MatrixMapConst = Eigen::Map<const Matrix>;


    Matrix complexToRealMatrix(const ComplexMatrix& complexMat, HostPrecision tolerance = 1e-6) {
        Matrix realMat(complexMat.rows(), complexMat.cols());

        for (int i = 0; i < complexMat.rows(); ++i) {
            for (int j = 0; j < complexMat.cols(); ++j) {
                if (std::abs(complexMat(i, j).imag()) > tolerance) {
                    throw std::runtime_error("Imaginary part exceeds tolerance at (" + std::to_string(i) + ", " + std::to_string(j) + ")");
                }
                realMat(i, j) = complexMat(i, j).real();  // Assign only the real part
            }
        }

        return realMat;
    }


template <typename Scalar>
void printScalar(const Scalar& value) {
    if constexpr (std::is_same_v<Scalar, ComplexType> || std::is_same_v<Scalar, std::complex<double>>) {
        std::cout << "(" << value.real() << ", " << value.imag() << ") ";
    } else {
        std::cout << value << " ";
    }
}
    
template <typename MatrixType>
void print(const MatrixType& mat, size_t n = std::numeric_limits<size_t>::max()) {
    using Scalar = typename MatrixType::Scalar;
    constexpr bool isVector = MatrixType::ColsAtCompileTime == 1;
    constexpr bool isRowMajor = MatrixType::IsRowMajor;
    if constexpr (isVector) {
        // Handle vectors
        std::cout << "Vector (" << mat.size() << "):\n";
        n = std::min(n, static_cast<size_t>(mat.size())); // Limit rows
        for (size_t i = 0; i < n; ++i) {
            printScalar(mat(i));  // Assuming `printScalar` is defined for the scalar type
            std::cout << " ";
        }
        std::cout << "\n";
    } else {
        // Handle matrices
        std::cout << "Matrix (" << mat.rows() << "x" << mat.cols() << ") - ";
        if constexpr (isRowMajor) {std::cout << "Row-major\n";}
        else {std::cout << "Col-major\n";}
        n = std::min(n, static_cast<size_t>(mat.rows())); // Limit rows
        for (size_t i = 0; i < n; ++i) {
            for (int j = 0; j < mat.cols(); ++j) {
                printScalar(mat(i, j));
                std::cout << " ";
            }
            std::cout << "\n";
        }
    }
}



#else
// Define std::vector types
    #include <vector>
    #include <span>
    using Vector = std::vector<HostPrecision>; // std::vector
    using ComplexVector = std::vector<std::complex<HostPrecision>>
    using Matrix = std::vector<Vector>; // 2D vector as matrix

    inline size_t rows(const Matrix& mat) {
        return mat.size(); // Return the number of rows
    }

    inline size_t cols(const Matrix& mat) {
        return mat.empty() ? 0 : mat[0].size(); // Return the number of columns
    }

    std::span<HostPrecision> slice(const Vector& vec, size_t i, size_t j) {
        if (i > j || j > vec.size()) {
            throw std::out_of_range("Invalid slice indices");
        }
        return std::span<HostPrecision>(vec.data() + i, j - 1);
    }

    std::span<Vector> slice(const Matrix& mat, size_t i, size_t j) {
        if (i > j || j > rows(mat)) {
            throw std::out_of_range("Invalid slice indices");
        }
        return std::span<Vector>(mat.data() + i, j - 1);
    }

    void printMatrix(const Matrix& mat) {
        std::cout << "Matrix (" << rows(mat) << "x" << cols(mat) << "):\n";
        for (size_t i = 0; i < rows(mat); ++i) {
            for (size_t j = 0; j < cols(mat); ++j) {
                std::cout << mat[i][j] << " ";
            }
            std::cout << "\n";
        }
    }


#endif

inline void mollify(Eigen::MatrixXcd& matrix, double tol=default_tol) noexcept {
    for (int i = 0; i < matrix.rows(); ++i) {
        for (int j = 0; j < matrix.cols(); ++j) {
            std::complex<double>& elem = matrix(i, j);
            if (std::abs(elem.real()) <= tol) {
                elem.real(0.0);
            }
            if (std::abs(elem.imag()) <= tol) {
                elem.imag(0.0);
            }
        }
    }
}



using Shape = std::pair<size_t, size_t>;
Shape shape(const Matrix& M) {
    #ifdef USE_EIGEN
        return Shape(M.rows(), M.cols());
    #else
        return Shape(rows(M), cols(M));
    #endif
}

inline HostPrecision norm(Vector& vec) {
    // Calculate the norm (L2 norm)
    HostPrecision norm = 0.0;
    for (HostPrecision value : vec) {
        norm += value * value;
    }
    return std::sqrt(norm);
}

//Normalizes a vector inplace
inline void normalize(Vector& vec) {
    HostPrecision Norm = norm(vec);
    for (float value : vec) {
        value /= Norm;
    }
}

Vector randVecGen(size_t N) {
    std::random_device rd;
    std::mt19937 gen(rd()); // Seed the random number generator
    std::normal_distribution<float> dist(0.0f, 1.0f); // Mean = 0, Stddev = 1

    // Initialize vector v0 with normal entries
    Vector v0(N);
    for (int i = 0; i < N; ++i) {
        v0[i] = dist(gen); // Fill vector with normal distributed values
    }
    return v0;
}

template <typename MatrixType>
MatrixType initMat(const size_t N) {
    using ScalarType = typename MatrixType::Scalar;

    MatrixType M;
    
    if (N > 10000) {
        M = MatrixType::Zero(N, N);
        
        if constexpr (std::is_same<ScalarType, std::complex<double>>::value || 
                      std::is_same<ScalarType, std::complex<float>>::value) {
            std::fill(M.data(), M.data() + M.size(), ScalarType(-1.0, -1.0));
        } else {
            std::fill(M.data(), M.data() + M.size(), ScalarType(-1.0));
        }
    } else {
        M = MatrixType::Random(N, N);
    }

    return M;
}

    

#endif // VECTOR_HPP