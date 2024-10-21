#ifndef ERRORMSG_HPP
#define ERRORMSG_HPP

#include <string>

constexpr std::string invalid_dims_msg(size_t a_cols, size_t a_rows, size_t b_cols, size_t b_rows) {
    return "Matrix dimensions are not compatible for multiplication: "
           + std::to_string(a_rows) + "x" + std::to_string(a_cols) + " * "
           + std::to_string(b_rows) + "x" + std::to_string(b_cols);
}

constexpr void CHECK_DIMS(const Matrix& A, const Vector& B) {
        #ifdef USE_EIGEN
            if (A.cols() != B.rows()) {
                throw std::invalid_argument(invalid_dims_msg(A.rows(), A.cols(), B.rows(), 1));
            }
        #else
            if (cols(A) != B.size()) {
                throw std::invalid_argument(invalid_dims_msg(rows(A), cols(A), B.size(), 1));
            }
        #endif
    }

#endif // ERRORMSG_HPP
