#ifndef SHIFT_TEST_HPP
#define SHIFT_TEST_HPP

#include <gtest/gtest.h>
#include "shift.hpp"
#include "arnoldi.hpp"

using MatType = Matrix;
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

    MatType M = initMat<MatType>(dims);
    
    ComplexKrylovPair q_h(RealKrylovIter<MatType>(M, max_iters, handle));

    // Test orthonormality of Q
    ASSERT_TRUE(isOrthonormal<ComplexMatrix>(q_h.Q));

    reduceArnoldiPair(q_h, basis_size, handle, solver_handle, resize_type::ZEROS);

    ASSERT_TRUE(isHessenberg<ComplexMatrix>(q_h.H));
    ASSERT_TRUE(isOrthonormal<ComplexMatrix>(q_h.Q.leftCols(basis_size), 1e-4));
}

#endif