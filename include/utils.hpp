#ifndef UTILS_HPP
#define UTILS_HPP

    #include "vector.hpp"
    #include <iostream>
    #include <cstring>
    #include <chrono>
    #include <cmath>
    #include <type_traits>

    // #define DBG_ORTHO


    #define LAPACKPP_CHECK(call) \
    do { \
        int64_t info = (call); \
        if (info != 0) { \
            std::cerr << "Error in " #call ": " << info << std::endl; \
            return info; \
        } \
    } while (0)

    template <typename M>
    void mollify(M& mat) {
        for (size_t i = 0; i < mat.rows(); ++i) {
            for (size_t j = 0; j < mat.cols(); ++j) {
                if (std::abs(mat(i, j).real()) < 1e-10) {
                    mat(i, j).real(0.0);
                }
                if (std::abs(mat(i, j).imag()) < 1e-10) {
                    mat(i, j).imag(0.0);
                }
            }
        }
    }


    template <typename MatrixType>
    inline bool isHessenberg(const MatrixType& mat, double tol = 1e-10) {
        using Scalar = typename MatrixType::Scalar;
        constexpr bool isComplex = std::is_same_v<Scalar, std::complex<double>> || std::is_same_v<Scalar, std::complex<float>>;

        const int rows = mat.rows();
        const int cols = mat.cols();

        bool ret_val = true;

        for (int i=2; i<rows; i++) {
            for (int j = 0; j < i - 1; j++) {
                if (std::norm(mat(i, j)) > tol * mat.norm()) {return false;}
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
        #ifdef DBG_ORTHO
        if (N <= 15) {
            std::cout << "Product of Q^H * Q: " << std::endl;
            mollify(product);
            print(product);
        }
        #endif
        bool ret = (product - MatrixType::Identity(Q.cols(), Q.cols())).norm() < tol;
        std::cout << (ret ? "SUCCESS" : "FAIL") << std::endl;
        return ret;
    }

    template <typename MatType>
    MatType gramSchmidtOrthonormal(const size_t& n) {
        MatType input = MatType::Random(n,n);
        MatType Q(n, n);
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

    template <typename MatType>
    MatType generateRandomSymmetricMatrix(size_t N) {
        static_assert(!std::is_same<typename MatType::Scalar, std::complex<double>>::value,
                        "generateRandomSymmetricMatrix only supports real-valued matrices.");
        MatType A = MatType::Random(N, N);       
        return A.template selfadjointView<Eigen::Upper>();
    }

    template <typename MatType>
    MatType generateRandomHermitianMatrix(size_t N) {
        static_assert(std::is_same_v<typename MatType::Scalar, ComplexType>, 
                        "HermitianEigenDecomp can only be used with matrices of complex type.");
        MatType A = Matrix::Random(N, N);
        return A.template selfadjointView<Eigen::Upper>();
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