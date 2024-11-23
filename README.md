## Overview

This project provides implementations of Arnoldi Iteration and Implicitly Restarted Arnoldi Method (IRAM) for eigenvalue computations of large arrays with the option for batched GPU matrix multiplication for acceleration of the iteration step. The main interface functions are **NaiveArnoldi** (arnoldi.hpp) and **IRAM** (IRAM.hpp), which are used to compute eigenvalues and eigenvectors of a given matrix given templated parameters for matrix size, maximum number of iterations and in the case of IRAM, max-basis-size and restart-size.

## To-Do

1. Project Cleanup
- Add run-time matrix-size, number of iterations specifications to both algorithms (current setup only due to path dependency, this will be fixed quickly by changing parameter specification in the shifting and iteration functions).
- Port to library with test executable instead of demo executable with main.cpp
- Cleanup use of vector libraries to allow non-Eigen use
- Make dependencies optional with build flags, write out naive cpu code for option not to use CUDA or if no device detected.

2. Algorithm Performance
- Numerical instability with creation of basis vectors for large combination of matrix dimensions/number of iterations before restarting.
- Numerical instability with batched cublas shifting approach. Memory logic works but shifted H matrix loses Hessenberg structure (likely due to repeated truncation of double multiplication in batched execution).

## Dependencies (As project stands)

- CUDA
- cuBLAS
- cuSOLVER
- Eigen 
- Google Test

## Building the Project

To build the project, run the following commands (project not yet ported to library so don't install yet):

```sh
mkdir build
cd build
cmake ..
make
```

## Usage

### NaiveArnoldi

The 

NaiveArnoldi

 function performs the Arnoldi Iteration to compute the eigenvalues and eigenvectors of a given matrix.

#### Example

```cpp
#include "arnoldi.hpp"
#include <cublas_v2.h>

int main() {
    cublasHandle_t handle;
    cublasCreate(&handle);

    constexpr size_t N = 1000; // Matrix size
    constexpr size_t max_iters = 100;

    ComplexMatrix M = ComplexMatrix::Random(N, N);
    ComplexEigenPairs ritzPairs = NaiveArnoldi<ComplexMatrix, N, N, max_iters>(M, handle);

    // Output the results
    std::cout << "Eigenvalues:\n" << ritzPairs.values << std::endl;
    std::cout << "Eigenvectors:\n" << ritzPairs.vectors << std::endl;

    cublasDestroy(handle);
    return 0;
}
```

### IRAM (Implicitly Restarted Arnoldi Method)

The 

IRAM

 function performs the Implicitly Restarted Arnoldi Method to compute the eigenvalues and eigenvectors of a given matrix.

#### Example

```cpp
#include "IRAM.hpp"
#include <cublas_v2.h>
#include <cusolverDn.h>

int main() {
    cublasHandle_t handle;
    cusolverDnHandle_t solver_handle;
    cublasCreate(&handle);
    cusolverDnCreate(&solver_handle);

    constexpr size_t N = 1000; // Matrix size
    constexpr size_t total_iters = 1000;
    constexpr size_t max_iters = 50;
    constexpr size_t basis_size = 10;

    ComplexMatrix M = ComplexMatrix::Random(N, N);
    HostPrecision matnorm = M.norm();
    for (auto& x : M.reshaped()) { x /= matnorm; } // For better numerical stability pass matrix w/ unit norm

    ComplexEigenPairs ritzPairs = IRAM<ComplexMatrix, N, total_iters, max_iters, basis_size>(M, handle, solver_handle);

    // Output the results
    std::cout << "Eigenvalues:\n" << ritzPairs.values << std::endl;
    std::cout << "Eigenvectors:\n" << ritzPairs.vectors << std::endl;

    cublasDestroy(handle);
    cusolverDnDestroy(solver_handle);
    return 0;
}
```


## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your changes.

## License

This project is licensed under the MIT License. See the LICENSE file for details.