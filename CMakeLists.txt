cmake_minimum_required(VERSION 3.16)

project(cpp_learning)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_VERBOSE_MAKEFILE ON)
set(DCMAKE_RULE_MESSAGES OFF)

set(CMAKE_CXX_STANDARD 20)

option(CPP_LEARNING_ENABLE_FINUFFT "Build CUDA-dependent FINUFFT example" OFF)

# version of the libraries to be used
set(EIGEN_DOWNLOAD_VERSION 3.4.0)
set(HIGHWAY_DOWNLOAD_VERSION 1.2.0)

set(CPM_DOWNLOAD_VERSION 0.42.0)
set(CPM_HASH_SUM "2020b4fc42dba44817983e06342e682ecfc3d2f484a581f11cc5731fbe4dce8a")


set(CPM_DOWNLOAD_LOCATION "${CMAKE_BINARY_DIR}/cmake/CPM_${CPM_DOWNLOAD_VERSION}.cmake")

# Expand relative path. This is important if the provided path contains a tilde (~)
get_filename_component(CPM_DOWNLOAD_LOCATION ${CPM_DOWNLOAD_LOCATION} ABSOLUTE)

file(DOWNLOAD
        https://github.com/cpm-cmake/CPM.cmake/releases/download/v${CPM_DOWNLOAD_VERSION}/CPM.cmake
        ${CPM_DOWNLOAD_LOCATION} EXPECTED_HASH SHA256=${CPM_HASH_SUM}
)

include(${CPM_DOWNLOAD_LOCATION})



#add_subdirectory(highway-experiments)
add_subdirectory(floating-points)
add_subdirectory(compilation-define-test)
add_subdirectory(sys-calls)
add_subdirectory(cache-latency)
add_subdirectory(eigen-experiments)
if(CPP_LEARNING_ENABLE_FINUFFT)
    add_subdirectory(finufft)
endif()
add_subdirectory(class-experiments)
add_subdirectory(kernelgen)
add_subdirectory(xsimd)
add_subdirectory(dispatch-test)