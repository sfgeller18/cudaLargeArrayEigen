# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.30

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/sfg18/projects/neurocpp

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/sfg18/projects/neurocpp/build

# Include any dependencies generated for this target.
include CMakeFiles/cuda_demo.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/cuda_demo.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/cuda_demo.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/cuda_demo.dir/flags.make

CMakeFiles/cuda_demo.dir/src/_test.cpp.o: CMakeFiles/cuda_demo.dir/flags.make
CMakeFiles/cuda_demo.dir/src/_test.cpp.o: /home/sfg18/projects/neurocpp/src/_test.cpp
CMakeFiles/cuda_demo.dir/src/_test.cpp.o: CMakeFiles/cuda_demo.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/sfg18/projects/neurocpp/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/cuda_demo.dir/src/_test.cpp.o"
	/usr/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/cuda_demo.dir/src/_test.cpp.o -MF CMakeFiles/cuda_demo.dir/src/_test.cpp.o.d -o CMakeFiles/cuda_demo.dir/src/_test.cpp.o -c /home/sfg18/projects/neurocpp/src/_test.cpp

CMakeFiles/cuda_demo.dir/src/_test.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/cuda_demo.dir/src/_test.cpp.i"
	/usr/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/sfg18/projects/neurocpp/src/_test.cpp > CMakeFiles/cuda_demo.dir/src/_test.cpp.i

CMakeFiles/cuda_demo.dir/src/_test.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/cuda_demo.dir/src/_test.cpp.s"
	/usr/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/sfg18/projects/neurocpp/src/_test.cpp -o CMakeFiles/cuda_demo.dir/src/_test.cpp.s

CMakeFiles/cuda_demo.dir/src/test.cu.o: CMakeFiles/cuda_demo.dir/flags.make
CMakeFiles/cuda_demo.dir/src/test.cu.o: CMakeFiles/cuda_demo.dir/includes_CUDA.rsp
CMakeFiles/cuda_demo.dir/src/test.cu.o: /home/sfg18/projects/neurocpp/src/test.cu
CMakeFiles/cuda_demo.dir/src/test.cu.o: CMakeFiles/cuda_demo.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/sfg18/projects/neurocpp/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CUDA object CMakeFiles/cuda_demo.dir/src/test.cu.o"
	/opt/cuda/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -MD -MT CMakeFiles/cuda_demo.dir/src/test.cu.o -MF CMakeFiles/cuda_demo.dir/src/test.cu.o.d -x cu -rdc=true -c /home/sfg18/projects/neurocpp/src/test.cu -o CMakeFiles/cuda_demo.dir/src/test.cu.o

CMakeFiles/cuda_demo.dir/src/test.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CUDA source to CMakeFiles/cuda_demo.dir/src/test.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/cuda_demo.dir/src/test.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CUDA source to assembly CMakeFiles/cuda_demo.dir/src/test.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

# Object files for target cuda_demo
cuda_demo_OBJECTS = \
"CMakeFiles/cuda_demo.dir/src/_test.cpp.o" \
"CMakeFiles/cuda_demo.dir/src/test.cu.o"

# External object files for target cuda_demo
cuda_demo_EXTERNAL_OBJECTS =

CMakeFiles/cuda_demo.dir/cmake_device_link.o: CMakeFiles/cuda_demo.dir/src/_test.cpp.o
CMakeFiles/cuda_demo.dir/cmake_device_link.o: CMakeFiles/cuda_demo.dir/src/test.cu.o
CMakeFiles/cuda_demo.dir/cmake_device_link.o: CMakeFiles/cuda_demo.dir/build.make
CMakeFiles/cuda_demo.dir/cmake_device_link.o: /opt/cuda/lib64/libcudart_static.a
CMakeFiles/cuda_demo.dir/cmake_device_link.o: /usr/lib/librt.a
CMakeFiles/cuda_demo.dir/cmake_device_link.o: /opt/cuda/lib64/libcublas.so
CMakeFiles/cuda_demo.dir/cmake_device_link.o: CMakeFiles/cuda_demo.dir/deviceLinkLibs.rsp
CMakeFiles/cuda_demo.dir/cmake_device_link.o: CMakeFiles/cuda_demo.dir/deviceObjects1.rsp
CMakeFiles/cuda_demo.dir/cmake_device_link.o: CMakeFiles/cuda_demo.dir/dlink.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/home/sfg18/projects/neurocpp/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CUDA device code CMakeFiles/cuda_demo.dir/cmake_device_link.o"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/cuda_demo.dir/dlink.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/cuda_demo.dir/build: CMakeFiles/cuda_demo.dir/cmake_device_link.o
.PHONY : CMakeFiles/cuda_demo.dir/build

# Object files for target cuda_demo
cuda_demo_OBJECTS = \
"CMakeFiles/cuda_demo.dir/src/_test.cpp.o" \
"CMakeFiles/cuda_demo.dir/src/test.cu.o"

# External object files for target cuda_demo
cuda_demo_EXTERNAL_OBJECTS =

cuda_demo: CMakeFiles/cuda_demo.dir/src/_test.cpp.o
cuda_demo: CMakeFiles/cuda_demo.dir/src/test.cu.o
cuda_demo: CMakeFiles/cuda_demo.dir/build.make
cuda_demo: /opt/cuda/lib64/libcudart_static.a
cuda_demo: /usr/lib/librt.a
cuda_demo: /opt/cuda/lib64/libcublas.so
cuda_demo: CMakeFiles/cuda_demo.dir/cmake_device_link.o
cuda_demo: CMakeFiles/cuda_demo.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/home/sfg18/projects/neurocpp/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Linking CXX executable cuda_demo"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/cuda_demo.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/cuda_demo.dir/build: cuda_demo
.PHONY : CMakeFiles/cuda_demo.dir/build

CMakeFiles/cuda_demo.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/cuda_demo.dir/cmake_clean.cmake
.PHONY : CMakeFiles/cuda_demo.dir/clean

CMakeFiles/cuda_demo.dir/depend:
	cd /home/sfg18/projects/neurocpp/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/sfg18/projects/neurocpp /home/sfg18/projects/neurocpp /home/sfg18/projects/neurocpp/build /home/sfg18/projects/neurocpp/build /home/sfg18/projects/neurocpp/build/CMakeFiles/cuda_demo.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles/cuda_demo.dir/depend
