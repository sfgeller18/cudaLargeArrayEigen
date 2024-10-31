#include <iostream>
#include <vector>
#include <algorithm>
#include <cassert>
#include "arnoldi.hpp"
#include "tests.hpp"

using MatType = MatrixColMajor;


//void testRitzPairs(const MatrixType& M, size_t max_iters, size_t basis_size, HostPrecision tol = 1e-5) {

int main(int argc, char* argv[]) {
    cublasHandle_t handle;
    cusolverDnHandle_t solver_handle;
    CHECK_CUBLAS(cublasCreate(&handle));
    CHECK_CUSOLVER(cusolverDnCreate(&solver_handle));

    const size_t& N = (argc > 1) ? std::stoi(argv[1]) : 10;
    const size_t& max_iters = (argc > 2) ? std::stoi(argv[2]) : 5;
    const size_t& basis_size = (argc > 3) ? std::stoi(argv[3]) : 2;

    const MatType H = generateRandomHessenbergMatrix<MatType>(N); //MatType::Random(N,N);
    ComplexKrylovPair q_h(RealKrylovIter<MatType>(H, max_iters, handle));

    EigenPairs H_eigensolution{};
    eigsolver<MatType>(H, H_eigensolution, N, matrix_type::HESSENBERG);
    testEigenpairs<MatType, EigenPairs>(H, H_eigensolution);
    const size_t k = basis_size / 2;
    reduceEvals(q_h, N, k, handle, solver_handle);

    CHECK_CUBLAS(cublasDestroy(handle));
    CHECK_CUSOLVER(cusolverDnDestroy(solver_handle));
    return 0;
}
