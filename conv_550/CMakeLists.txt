project("conv_550")
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

set(conv_src
#project550 files
${PROJECT_SOURCE_DIR}/src/convolutionKernel.cu
${PROJECT_SOURCE_DIR}/src/convolution.h
${PROJECT_SOURCE_DIR}/src/convolutionMethods.h
${PROJECT_SOURCE_DIR}/src/convolutionMethods.cpp

#nvidia files
${PROJECT_SOURCE_DIR}/src/convolutionSeparable_gold.cpp
${PROJECT_SOURCE_DIR}/src/convolutionSeparable.cu
${PROJECT_SOURCE_DIR}/src/convolutionSeparable_common.h

# outside libraries
${PROJECT_SOURCE_DIR}/src/stb_image_resize.h
${PROJECT_SOURCE_DIR}/src/stb_image_write.h
${PROJECT_SOURCE_DIR}/src/stb_image.h
)


#link_directories(
#    ${PROJECT_SOURCE_DIR}
#)


##########################################################################
#   Build GPU or CPU-only executable depending on cuda                   #
##########################################################################

include_directories( ${CUDA_INCLUDE_DIRS} )
include_directories(${CUDA_TOOLKIT_ROOT_DIR}/samples/common/inc)
include_directories(/home/lesia/NVIDIA_CUDA-8.0_Samples/common/inc)

#cuda_include_directories(${CMAKE_CURRENT_SOURCE_DIR})

set(CUDA_NVCC_FLAGS "-arch=sm_30" "--use_fast_math"  "-O3" "--ptxas-options=--verbose") # "-fmad=false" "-DTHRUST_DEVICE_BACKEND=THRUST_DEVICE_BACKEND_OMP"
add_definitions(-DCUDA_BUILD)

set(all_sources ${sources} ${cpu_sources} ${gpu_sources})
link_directories( ${CUDA_TOOLKIT_ROOT_DIR}/lib64/)


if(CUDA_FOUND)
    include_directories(${CUDA_TOOLKIT_ROOT_DIR}/samples/common/inc)
    add_definitions(-DCUDA_BUILD)
    cuda_add_executable(conv_550_demo ${PROJECT_SOURCE_DIR}/src/main.cpp ${conv_src} ${all_sources})
    #cuda_add_executable(conv_prof_nv ${PROJECT_SOURCE_DIR}/src/main_prof_nv.cpp ${conv_src} ${all_sources})
    #cuda_add_executable(conv_prof_compare_demo ${PROJECT_SOURCE_DIR}/src/mainForComparison.cpp ${conv_src} ${all_sources})
endif()





