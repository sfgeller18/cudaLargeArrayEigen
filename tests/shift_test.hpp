#ifndef SHIFT_TEST_HPP
#define SHIFT_TEST_HPP

#include <gtest/gtest.h>
#include "shift.hpp"
#include "arnoldi.hpp"

using MatType = Matrix;
using Traits = BasisTraits<MatType>;

constexpr size_t dims = 10;           // Example size for testing
constexpr size_t max_iters = 5;
constexpr size_t basis_size = 5;

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

    MatType M = MatType::Random(dims, dims);
    
    KrylovPair<typename MatType::Scalar> q_h = KrylovIter<MatType>(M, max_iters, handle);
    KrylovPair<ComplexType> q_h_complex = {ComplexMatrix(q_h.Q), ComplexMatrix(q_h.H), q_h.m};

    ASSERT_TRUE(isOrthonormal<OM>(q_h.Q));

    ComplexMatrix& H = q_h_complex.H;
    ComplexMatrix& Q = q_h_complex.Q;
    size_t& m = q_h_complex.m;

    reduceArnoldiPair(Q, H, Q.rows(), m, basis_size, handle, solver_handle, resize_type::ZEROS);

    ASSERT_TRUE(isHessenberg<OM>(q_h.H));
    ASSERT_TRUE(isOrthonormal<OM>(q_h.Q.leftCols(basis_size), 1e-4));
}



#endif