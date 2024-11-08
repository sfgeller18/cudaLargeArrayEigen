#include "interface.hpp"

int main() {
    Matrix M = Matrix::Random(1000, 1000);
    IRAMEigen<Matrix, 100, 50, 10, matrix_type::REGULAR> iram(M, 10);
}