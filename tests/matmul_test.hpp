#ifndef MATMUL_TESTS_HPP
#define MATMUL_TESTS_HPP

#include "matmul.hpp"
#include <gtest/gtest.h>

using RTMatType = MatrixRowMajor;

class MatMulTest : public ::testing::TestWithParam<std::tuple<size_t, size_t, bool>> {
protected:
    cublasHandle_t handle;

    void SetUp() override {
        cublasCreate(&handle);
    }

    void TearDown() override {
        cublasDestroy(handle);
    }
};

TEST_P(MatMulTest, MatmulInternal) {
    size_t Rows = std::get<0>(GetParam());
    size_t Cols = std::get<1>(GetParam());
    bool runHostCheck = std::get<2>(GetParam());

    RTMatType M = RTMatType::Random(Rows, Cols);
    Vector y = Vector::Random(Cols);
    Vector deviceResult = matmulHost<RTMatType>(M, y);

    if (runHostCheck) {
        Vector hostResult = M * y;
        ASSERT_TRUE(hostResult.isApprox(deviceResult, 1e-6)) 
            << "Results do not match for dimensions: " << Rows << "x" << Cols;
    }
}

// Define the parameter values for MatMulTest
INSTANTIATE_TEST_SUITE_P(
    MatMulTests,
    MatMulTest,
    ::testing::Values(
        std::make_tuple(4000, 5000, true),
        std::make_tuple(5000, 4000, true),
        std::make_tuple(40000, 50000, true)
    )
);

#endif // MATMUL_TESTS_HPP
