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

    const MatType M = MatType::Random(N, N);

    ComplexKrylovPair q_h(RealKrylovIter<MatType>(M, max_iters, handle));

    ComplexMatrix& H = q_h.H;
    ComplexMatrix& Q = q_h.Q;
    const size_t& m = q_h.m;
    EigenPairs H_eigensolution{};
    ComplexMatrix S = ComplexMatrix::Identity(m,m);

    eigsolver<ComplexMatrix>(H, H_eigensolution, H.cols(), matrix_type::HESSENBERG);
    computeShift(S, H, H_eigensolution.values, H.cols(), int(basis_size / 2), handle, solver_handle);
    isOrthonormal<ComplexMatrix>(S);
    H = S.adjoint() * H * S;
    Q = Q * S;

    // print(S);
    // print(Q);
    // print(H);

    CHECK_CUBLAS(cublasDestroy(handle));
    std::cout << "Destroyed cuBLAS handle." << std::endl;

    CHECK_CUSOLVER(cusolverDnDestroy(solver_handle));
    std::cout << "Destroyed cuSOLVER handle." << std::endl;

    return 0;
}
