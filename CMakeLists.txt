cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

# Set Clang/LLVM as the C++ and CUDA compiler before the project declaration
set(CMAKE_CXX_COMPILER "/usr/bin/clang++")
set(CMAKE_CUDA_COMPILER "/opt/cuda/bin/nvcc")

# Set project name and version
project(CudaDemo LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CUDA_STANDARD 23)

find_package(CUDA REQUIRED)
find_library(CUBLAS_LIBRARIES cublas HINTS ${CUDA_TOOLKIT_ROOT_DIR}/lib64)
find_library(CUSOLVER_LIBRARIES cusolver HINTS ${CUDA_TOOLKIT_ROOT_DIR}/lib64)
include_directories(${CUDA_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR}/include)
link_directories(${CUDA_LIBRARY_DIRS})

# Find LAPACK and BLAS
find_package(LAPACK REQUIRED)
find_library(LAPACKE_LIB NAMES lapacke REQUIRED)

find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})

include_directories("/opt/slate/include")  # Add include directories for LAPACK++ and BLAS++
link_directories("/opt/slate/lib")  # Add link directory for LAPACK++ and BLAS++

# Clang/CUDA warnings
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -diag-suppress 20012")

# Add source files from the src directory
file(GLOB SOURCES "src/*.cu" "src/*.cpp")

# Add executable target using sources from the src directory
add_executable(cuda_demo ${SOURCES})

# Set preprocessor defines
target_compile_definitions(cuda_demo PRIVATE USE_EIGEN PRECISION_DOUBLE)

# Specify the architecture (adjust to your GPU architecture)
set_target_properties(cuda_demo PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES 75  # Set the compute capability of your GPU
)

# Link against LAPACK++, BLAS++, and other libraries
target_link_libraries(cuda_demo PRIVATE ${GTEST_BOTH_LIBRARIES} ${CUDA_LIBRARIES} ${CUBLAS_LIBRARIES} ${CUSOLVER_LIBRARIES} -llapack -llapacke -lblas -llapackpp -lblaspp)

# Add a check to ensure Clang is correctly recognized for both CUDA and CXX
message(STATUS "CXX compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "CUDA compiler: ${CMAKE_CUDA_COMPILER}")

# Find Eigen (assuming it's installed in a standard location)
find_package(Eigen3 REQUIRED)
if(Eigen3_FOUND)
    include_directories(${EIGEN3_INCLUDE_DIR})
endif()

# Install rules
install(TARGETS cuda_demo
    RUNTIME DESTINATION bin  # Install the binary to bin/
)

install(DIRECTORY include/
    DESTINATION include  # Install headers to include/
    FILES_MATCHING PATTERN "*.h"
)

# Add uninstall target
if(NOT TARGET uninstall)
    configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake_uninstall.cmake.in"
        "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake"
        IMMEDIATE @ONLY
    )
    add_custom_target(uninstall
        COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake
    )
endif()
