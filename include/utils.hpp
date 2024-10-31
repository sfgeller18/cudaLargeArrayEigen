#ifndef UTILS_HPP
#define UTILS_HPP

    #include "vector.hpp"
    #include <iostream>
    #include <cstring>
    #include <chrono>
    #include <cmath>
    #include <type_traits>

    #define LAPACKPP_CHECK(call) \
    do { \
        int64_t info = (call); \
        if (info != 0) { \
            std::cerr << "Error in " #call ": " << info << std::endl; \
            return info; \
        } \
    } while (0)


    template <typename MatrixType>
    inline bool isHessenberg(const MatrixType& mat, double tol = 1e-10) {
        using Scalar = typename MatrixType::Scalar;
        constexpr bool isComplex = std::is_same_v<Scalar, std::complex<double>> || std::is_same_v<Scalar, std::complex<float>>;

        const int rows = mat.rows();
        const int cols = mat.cols();

        bool ret_val = true;

        for (int i = 2; i < rows; ++i) {          // Start from the 2nd subdiagonal
            for (int j = 0; j < i - 1 && j < cols; ++j) {
                double value = isComplex ? std::norm(mat(i, j)) : std::abs(mat(i, j));
                if (value > tol) {
                    ret_val = false;
                }
            }
        }
        // std::cout  << (ret_val ? "IS" : "IS NOT") << " HESSENBERG" <<std::endl;
        return true;
    }

    template <typename MatrixType>
    bool isOrthonormal(const MatrixType& Q, double tol = 1e-10) {
        using Scalar = typename MatrixType::Scalar;
        const size_t N = Q.cols();
        MatrixType product(N, N);

        if constexpr (std::is_same_v<Scalar, std::complex<double>>) {
            product = Q.adjoint() * Q;
        } else {
            product = Q.transpose() * Q;
        }

        bool ret = (product - MatrixType::Identity(Q.cols(), Q.cols())).norm() < tol;
        std::cout << (ret ? "SUCCESS" : "FAIL") << std::endl;
        return ret;
    }

    ComplexMatrix gramSchmidtOrthonormal(const size_t& n) {
            ComplexMatrix input = ComplexMatrix::Random(n,n);

            ComplexMatrix Q(n, n);
            
            // Copy first vector and normalize it
            Q.col(0) = input.col(0).normalized();
            
            for (int i = 1; i < n; ++i) {
                Q.col(i) = input.col(i);
                for (int j = 0; j < i; ++j) {
                    Q.col(i) -= (Q.col(j).adjoint() * input.col(i))[0] * Q.col(j);
                }        
                Q.col(i).normalize();
            }
            
            return Q;
        }


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

#endif