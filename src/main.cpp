#include "interface.hpp"

int main() {
    ComplexMatrix M = ComplexMatrix::Random(1000, 1000);
    IRAM<ComplexMatrix, 100, 50, 10> iram(M, 10);

    ComplexVector eigenvalues = iram.getEigenvalues();
    ComplexMatrix eigenvectors = iram.getEigenvectors();

    // print(eigenvalues);
    // print(eigenvectors);
    return 0;
    }