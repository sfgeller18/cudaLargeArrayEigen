// #include "tests.hpp"

// // Function to check if two vectors are approximately equal


// int main(int argc, char* argv[]) {
//     // Ensure N is passed from CLI
//     if (argc < 2) {
//         std::cerr << "Usage: " << argv[0] << " <size of vector N>" << std::endl;
//         return EXIT_FAILURE;
//     }

//     // Parse N from the command line
//     int N = std::atoi(argv[1]);

//     // Initialize matrix and vector
//     Matrix M(N, N);
//     Vector y(N);
//     std::cout << "Initializing matrix M and vector y" << std::endl;
//     std::memset(M.data(), -1, N * N * sizeof(HostPrecision));  // Assuming Matrix has a data() method returning float*
//     std::memset(y.data(), -1, N * sizeof(HostPrecision));  // Assuming Vector has a data() method returning float*
//     std::cout << "Matrix M and vector y initialized." << std::endl;

//     std::cout << "HEY";

//     Vector result_cpu = M * y;
//     Vector result_gpu = std::get<Vector>(matmul(M, y, CuRetType::HOST));

//     if (!check_correctness(result_cpu, result_gpu)) {
//         return EXIT_FAILURE; // If results differ
//     }


//     double avg_time = test_gpu_matmul_speed(M, y);
//     if (avg_time < 0.0) {
//         return EXIT_FAILURE; // Error occurred in timing
//     }


//     // Check correctness

//     return EXIT_SUCCESS;
// }
