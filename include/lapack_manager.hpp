#ifndef BLAS_LAPACK_HPP
#define BLAS_LAPACK_HPP

#include <lapacke.h>

#if defined(PRECISION_FLOAT)
    using lapackEigenPtr = int (*)(int, char, char, int, int, int, float*, int, float*, float*, float*, int);
#elif defined(PRECISION_DOUBLE)
    using lapackEigenPtr = int (*)(int, char, char, int, int, int, double*, int, double*, double*, double*, int);
#elif defined(PRECISION_FLOAT16)
    using lapackEigenPtr = void*; // Define as needed, but LAPACK may not support this.
    using lapackEvecPtr = void*;
#else
    #error "No precision defined! Please define PRECISION_FLOAT, PRECISION_DOUBLE, or PRECISION_FLOAT16."
#endif

constexpr lapackEigenPtr getEigenSolverFunction() {
#if defined(PRECISION_FLOAT)
    return LAPACKE_shseqr;
#elif defined(PRECISION_DOUBLE)
    return LAPACKE_dhseqr;
#elif defined(PRECISION_FLOAT16)
    // LAPACK does not natively support float16, handle as necessary
    return nullptr;
#else
    return nullptr;
#endif
}

constexpr lapackEigenPtr EigenSolver = getEigenSolverFunction();

#endif // BLAS_LAPACK_HPP