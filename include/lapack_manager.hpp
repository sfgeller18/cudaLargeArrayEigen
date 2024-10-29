#ifndef BLAS_LAPACK_HPP
#define BLAS_LAPACK_HPP

#include <lapack.hh>   // Main LAPACK++ header

#define LAPACK_CHECK(func_call) \
    { \
        int info = func_call; \
        if (info != 0) { \
            std::cerr << #func_call " failed with error code: " << info << std::endl; \
            return {}; \
        } \
    }

#if defined(PRECISION_FLOAT)
    using lapackEigenPtr = int (*)(int, char, char, int, int, int, std::complex<float>*, int, std::complex<float>*, std::complex<float>*, int);
#elif defined(PRECISION_DOUBLE)
    using lapackEigenPtr = int (*)(int, char, char, int, int, int, std::complex<double>*, int, std::complex<double>*, std::complex<double>*, int);
#elif defined(PRECISION_FLOAT16)
    using lapackEigenPtr = void*; // Not supported for float16
#else
    #error "No precision defined! Please define PRECISION_FLOAT, PRECISION_DOUBLE, or PRECISION_FLOAT16."
#endif


// int matrix_layout, char job, char compz, lapack_int n,
//                            lapack_int ilo, lapack_int ihi,
//                            lapack_complex_double* h, lapack_int ldh,
//                            lapack_complex_double* w, lapack_complex_double* z,
//                            lapack_int ldz 

constexpr auto getEigenSolverFunction() {
#if defined(PRECISION_FLOAT)
    return [](int matrix_layout, char job, char compz, int n, int ilo, int ihi,
              std::complex<float>* h, int ldh, std::complex<float>* w,
              std::complex<float>* z, int ldz) -> int {
        return LAPACKE_chseqr(matrix_layout, job, compz, n, ilo, ihi,
                              reinterpret_cast<lapack_complex_float*>(h), ldh,
                              reinterpret_cast<lapack_complex_float*>(w),
                              reinterpret_cast<lapack_complex_float*>(z), ldz);
    };
#elif defined(PRECISION_DOUBLE)
    return [](int matrix_layout, char job, char compz, int n, int ilo, int ihi,
              std::complex<double>* h, int ldh, std::complex<double>* w,
              std::complex<double>* z, int ldz) -> int {
        return LAPACKE_zhseqr(matrix_layout, job, compz, n, ilo, ihi,
                              reinterpret_cast<lapack_complex_double*>(h), ldh,
                              reinterpret_cast<lapack_complex_double*>(w),
                              reinterpret_cast<lapack_complex_double*>(z), ldz);
    };
#elif defined(PRECISION_FLOAT16)
    return nullptr;
#else
    return nullptr;
#endif
}

constexpr lapackEigenPtr ComplexEigenSolver = getEigenSolverFunction();

#endif // BLAS_LAPACK_HPP