#ifndef SHIFT_TEST_HPP
#define SHIFT_TEST_HPP

#include <gtest/gtest.h>
#include "shift.hpp"
#include "arnoldi.hpp"
#include <ctime>

using MatType = Matrix;
using Traits = BasisTraits<MatType>;

constexpr size_t dims = 1000;           // Example size for testing
constexpr size_t max_iters = 100;
constexpr size_t basis_size = 10;

class CudaTest : public ::testing::Test {
protected:
    cublasHandle_t handle;
    cusolverDnHandle_t solver_handle;

    void SetUp() override {
        cublasCreate(&handle);
        cusolverDnCreate(&solver_handle);
    }

    void TearDown() override {
        cublasDestroy(handle);
        cusolverDnDestroy(solver_handle);
    }
};



TEST_F(CudaTest, KrylovIterationAndArnoldiReduction) {
    using OM = typename Traits::OM;
    std::srand(std::time(nullptr));
    MatType M = MatType::Random(dims, dims);
    
    KrylovPair<typename MatType::Scalar> q_h = KrylovIter<MatType>(M, max_iters, handle);

    ASSERT_TRUE(isOrthonormal<OM>(q_h.Q));

    MatType& H = q_h.H;
    MatType& Q = q_h.Q;
    size_t& m = q_h.m;

    ComplexMatrix H_square(max_iters, max_iters);
    ComplexMatrix Q_block(dims, max_iters);

    reduceArnoldiPairInternal<MatType, dims, max_iters>(Q, H, basis_size, handle, solver_handle, H_square, Q_block);

    std::cout << "H: " << H.topLeftCorner(10,10) << std::endl;

    ASSERT_TRUE(isHessenberg<OM>(q_h.H));
    ASSERT_TRUE(isOrthonormal<OM>(q_h.Q.leftCols(basis_size), 1e-4));
}






#endif