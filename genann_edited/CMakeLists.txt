project("nn_550")
cmake_minimum_required(VERSION 2.8)

if( NOT CMAKE_BUILD_TYPE )
   set( CMAKE_BUILD_TYPE Release )
endif()

add_definitions(-std=c++11)

# find packages with modules
find_package(CUDA REQUIRED)


include_directories(
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/src
)

set(nn_src
${PROJECT_SOURCE_DIR}/runmnist.cu
${PROJECT_SOURCE_DIR}/mnist.cu
${PROJECT_SOURCE_DIR}/mnist.cuh

# outside libraries
${PROJECT_SOURCE_DIR}/genann.h
#${PROJECT_SOURCE_DIR}/genann.c
${PROJECT_SOURCE_DIR}/genann_backprop_edits.cu
)

#link_directories(
#    ${PROJECT_SOURCE_DIR}
#)


##########################################################################
#   Build GPU or CPU-only executable depending on cuda                   #
##########################################################################

include_directories( ${CUDA_INCLUDE_DIRS} )
include_directories(${CUDA_TOOLKIT_ROOT_DIR}/samples/common/inc)
#cuda_include_directories(${CMAKE_CURRENT_SOURCE_DIR})

set(CUDA_NVCC_FLAGS "-arch=sm_30" "--use_fast_math"  "-O3" "--ptxas-options=--verbose") # "-fmad=false" "-DTHRUST_DEVICE_BACKEND=THRUST_DEVICE_BACKEND_OMP"
add_definitions(-DCUDA_BUILD)

set(all_sources ${sources} ${cpu_sources} ${gpu_sources})
link_directories( ${CUDA_TOOLKIT_ROOT_DIR}/lib64/)


if(CUDA_FOUND)
    include_directories(${CUDA_TOOLKIT_ROOT_DIR}/samples/common/inc)
    add_definitions(-DCUDA_BUILD)
    cuda_add_executable(nn_src_demo ${nn_src} ${all_sources})
endif()





