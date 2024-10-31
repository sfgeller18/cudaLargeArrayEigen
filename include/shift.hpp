#ifndef SHIFT_HPP
#define SHIFT_HPP
    //TO-DO, write cuda kernel to simultaneously compute QR decomp of p = k - m shifted Hessenberg matrices.
    //Just setup blocks running Householder on each shifted matrix

    #include <vector>
    #include <algorithm>
    #include <complex>
    #include <numeric>
    #include <cstddef>
    #include "vector.hpp"
    #include "cuda_manager.hpp"
    #include "utils.hpp"

    #include <iostream>


    inline DeviceComplexType toDeviceComplex(const std::complex<double>& c) {
        return make_cuDoubleComplex(c.real(), c.imag());
    }

    // #define DEVICE_SHIFT

    using MatrixType = MatrixColMajor;

    inline int constructSMatrix(cusolverDnHandle_t solver_handle,
                            cublasHandle_t blas_handle,
                            const std::vector<DeviceComplexType*>& h_Aarray, 
                            const std::vector<DeviceComplexType*>& h_Tauarray,
                            const int m,
                            const int n,
                            const int lda,
                            ComplexMatrix& S,
                            const int batch_count) {
        // Allocate device memory for Q matrix and result
        DeviceComplexType* d_Q;
        DeviceComplexType* d_S;
        DeviceComplexType* d_temp; // For intermediate results
        int* d_info;

        CHECK_CUDA(cudaMalloc(&d_Q, m * m * sizeof(DeviceComplexType)));
        CHECK_CUDA(cudaMalloc(&d_S, m * m * sizeof(DeviceComplexType)));
        CHECK_CUDA(cudaMalloc(&d_temp, m * m * sizeof(DeviceComplexType)));
        CHECK_CUDA(cudaMalloc(&d_info, sizeof(int)));

        // Buffer for Zungqr
        DeviceComplexType* d_work;
        int lwork;
        CHECK_CUSOLVER(cusolverDnZungqr_bufferSize(
            solver_handle, m, m, std::min(m,n),
            d_Q, lda,
            h_Tauarray[0],
            &lwork));
        CHECK_CUDA(cudaMalloc(&d_work, lwork * sizeof(DeviceComplexType)));

        // Constants for cublas
        const DeviceComplexType alpha = make_cuDoubleComplex(1.0, 0.0);
        const DeviceComplexType beta = make_cuDoubleComplex(0.0, 0.0);
        ComplexMatrix Q(m,n);
        for (int batch_idx = 0; batch_idx < batch_count; batch_idx++) {
            // Copy the QR factored matrix to d_Q
            CHECK_CUDA(cudaMemcpy(d_Q, h_Aarray[batch_idx], 
                                m * n * sizeof(DeviceComplexType),
                                cudaMemcpyDeviceToDevice));
            
            // Construct Q matrix from the QR factorization
            CHECK_CUSOLVER(cusolverDnZungqr(
                solver_handle, m, m, std::min(m,n),
                d_Q, lda,
                h_Tauarray[batch_idx],
                d_work, lwork,
                d_info));

        //         // Copy result back to host
        //         CHECK_CUDA(cudaMemcpy(Q.data(), d_Q,
        //                          m * m * sizeof(DeviceComplexType),
        //                          cudaMemcpyDeviceToHost));
        //        if (batch_idx == 0) {
        //             S = Q;
        //         } else {
        //             S = S * Q;
        //         }
            
        //     // Free workspace for next iteration
        // }

            if (batch_idx == 0) {
                // For first iteration, just copy Q to S
                CHECK_CUDA(cudaMemcpy(d_S, d_Q, 
                                    m * m * sizeof(DeviceComplexType),
                                    cudaMemcpyDeviceToDevice));
            } else {
                // S = S * Q
                // First copy current S to temp
                CHECK_CUDA(cudaMemcpy(d_temp, d_S,
                                    m * m * sizeof(DeviceComplexType),
                                    cudaMemcpyDeviceToDevice));
                
                // Perform matrix multiplication: d_S = d_temp * d_Q
                CHECK_CUBLAS(cublasZgemm(blas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                    m, m, m,
                                    &alpha,
                                    d_temp, m,
                                    d_Q, m,
                                    &beta,
                                    d_S, m));
            }
        }

        // Copy final result back to host
        CHECK_CUDA(cudaMemcpy(S.data(), d_S,
                            m * m * sizeof(DeviceComplexType),
                            cudaMemcpyDeviceToHost));
        
        // Clean up
        CHECK_CUDA(cudaFree(d_Q));
        CHECK_CUDA(cudaFree(d_S));
        CHECK_CUDA(cudaFree(d_temp));
        CHECK_CUDA(cudaFree(d_info));
        CHECK_CUDA(cudaFree(d_work));

        return 0;
    }

    int computeShift(ComplexMatrix& S,
                    const ComplexMatrix& complex_M, 
                    const ComplexVector& eigenvalues,
                    const size_t& N,
                    const size_t& m,
                    cublasHandle_t& handle,
                    cusolverDnHandle_t& solver_handle) {
        assert(m < N);
        isHessenberg(complex_M);
        const size_t N_squared = N * N;
        const size_t batch_count = N - m;
        std::cout << N_squared << ", " << batch_count << std::endl;

        // Allocate device memory for matrices and tau
        DeviceComplexType* d_matrices;
        DeviceComplexType* d_tau;

        CHECK_CUDA(cudaMalloc(&d_matrices, batch_count * N_squared * sizeof(DeviceComplexType)));
        CHECK_CUDA(cudaMalloc(&d_tau, batch_count * N * sizeof(DeviceComplexType)));
        std::cout << "HEY" << std::endl;

        // Allocate array of pointers
        DeviceComplexType** d_Aarray;
        DeviceComplexType** d_Tauarray;
        CHECK_CUDA(cudaMalloc(&d_Aarray, batch_count * sizeof(DeviceComplexType*)));
        CHECK_CUDA(cudaMalloc(&d_Tauarray, batch_count * sizeof(DeviceComplexType*)));
        std::cout << "HEY" << std::endl;

        std::vector<size_t> indices(batch_count);
        std::iota(indices.begin(), indices.end(), 0);
        std::sort(indices.begin(), indices.end(),
                [&eigenvalues](size_t i1, size_t i2) { return std::norm(eigenvalues[i1]) < std::norm(eigenvalues[i2]); });
        std::cout << "HEY" << std::endl;

        ComplexVector smallest_eigenvalues(batch_count);
        for (size_t i = 0; i < batch_count; ++i) {
            smallest_eigenvalues[i] = eigenvalues[indices[i]];
        }
        std::cout << "HEY" << std::endl;

        // Prepare matrices on host
        std::vector<DeviceComplexType> h_matrices(batch_count * N_squared);
        for (int k = 0; k < batch_count; k++) {
            for (int j = 0; j < N; j++) {
                for (int i = 0; i < N; i++) {
                    size_t idx = k * N_squared + j * N + i;
                    if (i == j) {
                        h_matrices[idx] = toDeviceComplex(complex_M(j,i) - smallest_eigenvalues[k]);
                    } else {
                        h_matrices[idx] = toDeviceComplex(complex_M(j,i));
                    }
                }
            }
        }
        std::cout << "HEY" << std::endl;

        // Copy matrices to device
        CHECK_CUDA(cudaMemcpy(d_matrices, h_matrices.data(), 
                            batch_count * N_squared * sizeof(DeviceComplexType), 
                            cudaMemcpyHostToDevice));
        std::cout << "HEY" << std::endl;

        // Set up arrays of pointers
        std::vector<DeviceComplexType*> h_Aarray(batch_count);
        std::vector<DeviceComplexType*> h_Tauarray(batch_count);
        for (int i = 0; i < batch_count; i++) {
            h_Aarray[i] = d_matrices + i * N_squared;
            h_Tauarray[i] = d_tau + i * N;
        }
        std::cout << "HEY" << std::endl;

        // Copy pointer arrays to device
        CHECK_CUDA(cudaMemcpy(d_Aarray, h_Aarray.data(), batch_count * sizeof(DeviceComplexType*), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_Tauarray, h_Tauarray.data(), batch_count * sizeof(DeviceComplexType*), cudaMemcpyHostToDevice));
        std::cout << "HEY" << std::endl;

        // Perform batched QR factorization
        int info;
        CHECK_CUBLAS(cublasZgeqrfBatched(handle, N, N, d_Aarray, N, d_Tauarray, &info, batch_count));
        std::cout << "HEY" << std::endl;

        // Pass the device array pointer and host tau array
        constructSMatrix(solver_handle, handle, h_Aarray, h_Tauarray, N, N, N, S, batch_count);
        std::cout << "HEY" << std::endl;
        
        isOrthonormal<ComplexMatrix>(S);
        ComplexMatrix prod = S.adjoint() * complex_M * S;
        std::cout << "HEY" << std::endl;

        // Clean up
        CHECK_CUDA(cudaFree(d_matrices));
        CHECK_CUDA(cudaFree(d_tau));
        CHECK_CUDA(cudaFree(d_Aarray));
        CHECK_CUDA(cudaFree(d_Tauarray));
        std::cout << "HEY" << std::endl;

        return 0;
    }

    //Modifies Krylov Pair in place
    int reduceEvals(ComplexKrylovPair& pair, const size_t& N, const size_t& k, cublasHandle_t& handle, cusolverDnHandle_t solver_handle) {
        ComplexMatrix& H = pair.H;
        ComplexMatrix& Q = pair.Q;
        EigenPairs eigpairs{};
        eigsolver<ComplexMatrix>(H, eigpairs, N, matrix_type::HESSENBERG);
        ComplexMatrix S(N,N);
        std::cout << k << std::endl;

        // Create cublas handle and perform QR factorization

        computeShift(S, H, eigpairs.values, N, k, handle, solver_handle);
        H = S * H * S.adjoint();
        isOrthonormal<ComplexMatrix>(S);

    
        // print(Q);
        // print(H);
        return 0;
    }

    // int main() {
    //     const size_t N = 5;
    //     const size_t k = 2;
    //     ComplexVector evals(N);
    //     ComplexMatrix evecs(N,N);
    //     ComplexMatrix H = generateRandomHessenbergMatrix<ComplexMatrix>(N);
    //     ComplexMatrix V = gramSchmidtOrthonormal(N);

    //     eigsolver(H, evals, evecs);
    //     reduceEvals(H, V, N, k);
    //     print(V);
    // }

#endif
