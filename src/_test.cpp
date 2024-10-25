#include <cassert>
#include <cmath>
#include "eigenSolver.hpp"
#include "vector.hpp"
#include "tests.hpp"







// Example usage with test cases



using MatrixType = MatrixColMajor;


int main(int argc, char* argv[]) {
    return eigSolverTest<MatrixType>(argc, argv);
}