#include <gtest/gtest.h>
#include "cuda_manager.hpp"
#include "utils.hpp"

constexpr size_t test_m = 10;
constexpr size_t test_n = 10;

class CublasTest : public ::testing::Test {
protected:
    cublasHandle_t handle;

    void SetUp() override {
        // Create cuBLAS handle
        cublasCreate(&handle);
    }

    void TearDown() override {
        // Destroy cuBLAS handle
        cublasDestroy(handle);
    }
};

TEST_F(CublasTest, GemvTest) {
    // Using Eigen for random matrix generation
    Matrix A = Matrix::Random(test_m, test_n);
    Eigen::VectorXd x = Eigen::VectorXd::Random(test_n);
    Eigen::VectorXd y = Eigen::VectorXd::Zero(test_m);

    Eigen::VectorXd eigen_result = A * x;

    DevicePrecision alpha = 1.0;
    DevicePrecision beta = 0.0;

    DevicePrecision *d_A, *d_x, *d_y;
    cudaMalloc(&d_A, A.size() * sizeof(DevicePrecision));
    cudaMalloc(&d_x, x.size() * sizeof(DevicePrecision));
    cudaMalloc(&d_y, y.size() * sizeof(DevicePrecision));

    cudaMemcpy(d_A, A.data(), A.size() * sizeof(DevicePrecision), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x.data(), x.size() * sizeof(DevicePrecision), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y.data(), y.size() * sizeof(DevicePrecision), cudaMemcpyHostToDevice);

    // Call gemv function
    auto status = cublas::gemv<DevicePrecision>(handle, CUBLAS_OP_N, test_m, test_n, &alpha, d_A, test_m, d_x, 1, &beta, d_y, 1);

    ASSERT_EQ(status, CUBLAS_STATUS_SUCCESS);

    // Copy result back to host
    cudaMemcpy(y.data(), d_y, y.size() * sizeof(DevicePrecision), cudaMemcpyDeviceToHost);

    ASSERT_LE((y - eigen_result).norm(), 1e-6);

    cudaFree(d_A);
    cudaFree(d_x);
    cudaFree(d_y);
}

// Test for norm function
TEST_F(CublasTest, NormTest) {
    int N = 3;
    // Using random vector for testing
    Eigen::VectorXd d_result = Eigen::VectorXd::Random(N);
    DevicePrecision norms = 0.0;
    DevicePrecision *d_result_dev;

    cudaMalloc(&d_result_dev, d_result.size() * sizeof(DevicePrecision));
    cudaMemcpy(d_result_dev, d_result.data(), d_result.size() * sizeof(DevicePrecision), cudaMemcpyHostToDevice);

    // Call norm function
    auto status = cublas::norm<DevicePrecision>(handle, N, d_result_dev, 1, &norms);

    ASSERT_EQ(status, CUBLAS_STATUS_SUCCESS);

    EXPECT_NEAR(norms, d_result.norm(), 1e-6);  // Expecting norm to match Eigen norm

    cudaFree(d_result_dev);
}

// Test for batchedQR function
TEST_F(CublasTest, BatchedQRTest) {
    int m = 3, n = 3, batch_count = 1;
    std::vector<DevicePrecision*> d_Aarray(batch_count, nullptr);  // Using nullptr for test case
    std::vector<DevicePrecision*> d_Tauarray(batch_count, nullptr);  // Same here
    int *d_info = nullptr;  // No memory allocated here for test

    // Call batchedQR function
    auto status = cublas::batchedQR<DevicePrecision>(handle, m, n, d_Aarray.data(), m, d_Tauarray.data(), d_info, batch_count);

    ASSERT_EQ(status, CUBLAS_STATUS_INVALID_VALUE);  // Expecting error since no memory was allocated
}
