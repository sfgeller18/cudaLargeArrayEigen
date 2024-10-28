//TO-DO, write cuda kernel to simultaneously compute QR decomp of p = k - m shifted Hessenberg matrices.
//Just setup blocks running Householder on each shifted matrix

#include <vector>
#include <algorithm>
#include <complex>
#include <numeric>
#include <cstddef>
#include "vector.hpp"
#include "cuda_manager.hpp"


inline DeviceComplexType toDeviceComplex(const std::complex<double>& c) {
    return make_cuDoubleComplex(c.real(), c.imag());
}

