# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 2.8

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list

# Suppress display of executed commands.
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
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/home1/rachel/Documents/Architecture/final_project/cuANN_edited

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/home1/rachel/Documents/Architecture/final_project/cuANN_edited/build

# Include any dependencies generated for this target.
include CMakeFiles/diabetes.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/diabetes.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/diabetes.dir/flags.make

CMakeFiles/cuda_compile.dir/samples/./cuda_compile_generated_diabetes.cu.o: CMakeFiles/cuda_compile.dir/samples/cuda_compile_generated_diabetes.cu.o.depend
CMakeFiles/cuda_compile.dir/samples/./cuda_compile_generated_diabetes.cu.o: CMakeFiles/cuda_compile.dir/samples/cuda_compile_generated_diabetes.cu.o.cmake
CMakeFiles/cuda_compile.dir/samples/./cuda_compile_generated_diabetes.cu.o: ../samples/diabetes.cu
	$(CMAKE_COMMAND) -E cmake_progress_report /home/home1/rachel/Documents/Architecture/final_project/cuANN_edited/build/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold "Building NVCC (Device) object CMakeFiles/cuda_compile.dir/samples/./cuda_compile_generated_diabetes.cu.o"
	cd /home/home1/rachel/Documents/Architecture/final_project/cuANN_edited/build/CMakeFiles/cuda_compile.dir/samples && /usr/bin/cmake -E make_directory /home/home1/rachel/Documents/Architecture/final_project/cuANN_edited/build/CMakeFiles/cuda_compile.dir/samples/.
	cd /home/home1/rachel/Documents/Architecture/final_project/cuANN_edited/build/CMakeFiles/cuda_compile.dir/samples && /usr/bin/cmake -D verbose:BOOL=$(VERBOSE) -D build_configuration:STRING= -D generated_file:STRING=/home/home1/rachel/Documents/Architecture/final_project/cuANN_edited/build/CMakeFiles/cuda_compile.dir/samples/./cuda_compile_generated_diabetes.cu.o -D generated_cubin_file:STRING=/home/home1/rachel/Documents/Architecture/final_project/cuANN_edited/build/CMakeFiles/cuda_compile.dir/samples/./cuda_compile_generated_diabetes.cu.o.cubin.txt -P /home/home1/rachel/Documents/Architecture/final_project/cuANN_edited/build/CMakeFiles/cuda_compile.dir/samples/cuda_compile_generated_diabetes.cu.o.cmake

# Object files for target diabetes
diabetes_OBJECTS =

# External object files for target diabetes
diabetes_EXTERNAL_OBJECTS = \
"/home/home1/rachel/Documents/Architecture/final_project/cuANN_edited/build/CMakeFiles/cuda_compile.dir/samples/./cuda_compile_generated_diabetes.cu.o"

diabetes: CMakeFiles/cuda_compile.dir/samples/./cuda_compile_generated_diabetes.cu.o
diabetes: CMakeFiles/diabetes.dir/build.make
diabetes: /usr/local/cuda/lib64/libcudart.so
diabetes: libcuANN.so
diabetes: /usr/local/cuda/lib64/libcudart.so
diabetes: /usr/lib/x86_64-linux-gnu/libboost_system.so
diabetes: /usr/lib/x86_64-linux-gnu/libboost_thread.so
diabetes: /usr/lib/x86_64-linux-gnu/libboost_serialization.so
diabetes: CMakeFiles/diabetes.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable diabetes"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/diabetes.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/diabetes.dir/build: diabetes
.PHONY : CMakeFiles/diabetes.dir/build

CMakeFiles/diabetes.dir/requires:
.PHONY : CMakeFiles/diabetes.dir/requires

CMakeFiles/diabetes.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/diabetes.dir/cmake_clean.cmake
.PHONY : CMakeFiles/diabetes.dir/clean

CMakeFiles/diabetes.dir/depend: CMakeFiles/cuda_compile.dir/samples/./cuda_compile_generated_diabetes.cu.o
	cd /home/home1/rachel/Documents/Architecture/final_project/cuANN_edited/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/home1/rachel/Documents/Architecture/final_project/cuANN_edited /home/home1/rachel/Documents/Architecture/final_project/cuANN_edited /home/home1/rachel/Documents/Architecture/final_project/cuANN_edited/build /home/home1/rachel/Documents/Architecture/final_project/cuANN_edited/build /home/home1/rachel/Documents/Architecture/final_project/cuANN_edited/build/CMakeFiles/diabetes.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/diabetes.dir/depend

