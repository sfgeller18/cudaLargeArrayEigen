#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include <algorithm>
#include <cassert>
#include "tests.hpp"

using ComplexMatrix = Eigen::MatrixXcd;
using ComplexVector = Eigen::VectorXcd;

// Construct S matrix from QR factored matrices in a batch
int constructSMatrix(const std::vector<ComplexMatrix>& h_Aarray,
                     const std::vector<ComplexVector>& h_Tauarray,
                     int m, int n, ComplexMatrix& S, int batch_count) {

    ComplexMatrix Q(m, n);
    for (int batch_idx = 0; batch_idx < batch_count; batch_idx++) {
        Q = h_Aarray[batch_idx];
        // Perform QR factorization
        Eigen::HouseholderQR<ComplexMatrix> qr(Q);
        Q = qr.householderQ(); // Construct Q from the factorization
        
        // Accumulate product of Q matrices
        if (batch_idx == 0) {
            S = Q;
        } else {
            S = S * Q;
        }
    }
    return 0;
}

int computeShift(ComplexMatrix& S,
                 const ComplexMatrix& complex_M,
                 const ComplexVector& eigenvalues,
                 size_t N, size_t m) {
    assert(complex_M.rows() == N && complex_M.cols() == N);
    
    size_t batch_count = N - m;
    std::vector<size_t> indices(batch_count);
    std::iota(indices.begin(), indices.end(), 0);

    std::sort(indices.begin(), indices.end(),
              [&eigenvalues](size_t i1, size_t i2) { return std::norm(eigenvalues[i1]) < std::norm(eigenvalues[i2]); });

    ComplexVector smallest_eigenvalues(batch_count);
    for (size_t i = 0; i < batch_count; ++i) {
        smallest_eigenvalues[i] = eigenvalues[indices[i]];
    }

    // Prepare matrices on host
    std::vector<ComplexMatrix> h_Aarray(batch_count, ComplexMatrix(N, N));
    for (int k = 0; k < batch_count; k++) {
        ComplexMatrix M = complex_M;
        for (int i = 0; i < N; i++) {
            M(i, i) -= smallest_eigenvalues[k];
        }
        h_Aarray[k] = M;
    }

    std::vector<ComplexVector> h_Tauarray(batch_count, ComplexVector(N));

    // Perform QR factorization on each matrix
    for (int i = 0; i < batch_count; i++) {
        Eigen::HouseholderQR<ComplexMatrix> qr(h_Aarray[i]);
        h_Aarray[i] = qr.householderQ(); // Q matrix
        h_Tauarray[i] = qr.matrixQR().diagonal(); // Tau vector for each QR
    }

    // Construct the shift matrix S
    constructSMatrix(h_Aarray, h_Tauarray, N, N, S, batch_count);
    isOrthonormal<ComplexMatrix>(S);    
    ComplexMatrix prod = S.adjoint() * complex_M * S;
    return 0;
}

int reduceEvals(ComplexMatrix& H, ComplexMatrix& Q, size_t N, size_t k) {
    print(Q);
    print(H);
    
    ComplexVector evals(N);
    ComplexMatrix evecs(N, N);

    eigsolver(H, evals, evecs);

    ComplexMatrix S(N, N);
    computeShift(S, H, evals, N, k);

    ComplexMatrix temp = S * H * S.adjoint();
    return 0;
}

int main() {
    const size_t N = 5;
    const size_t k = 2;

    ComplexVector evals(N);
    ComplexMatrix evecs(N, N);
    ComplexMatrix H = generateRandomHessenbergMatrix<ComplexMatrix>(N);
    ComplexMatrix V = gramSchmidtOrthonormal(N);

    eigsolver(H, evals, evecs);
    reduceEvals(H, V, N, k);

    print(V);
    return 0;
}
