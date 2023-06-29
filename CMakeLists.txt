cmake_minimum_required(VERSION 3.16)

project(cpp_learning)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_VERBOSE_MAKEFILE ON)
set(DCMAKE_RULE_MESSAGES OFF)

set(CMAKE_CXX_STANDARD 20)

# version of the libraries to be used
set(CPM_DOWNLOAD_VERSION 0.38.1)
set(EIGEN_DOWNLOAD_VERSION 3.4.0)
set(HIGHWAY_DOWNLOAD_VERSION 1.0.4)


if (CPM_SOURCE_CACHE)
    set(CPM_DOWNLOAD_LOCATION "${CPM_SOURCE_CACHE}/cpm/CPM_${CPM_DOWNLOAD_VERSION}.cmake")
elseif (DEFINED ENV{CPM_SOURCE_CACHE})
    set(CPM_DOWNLOAD_LOCATION "$ENV{CPM_SOURCE_CACHE}/cpm/CPM_${CPM_DOWNLOAD_VERSION}.cmake")
else ()
    set(CPM_DOWNLOAD_LOCATION "${CMAKE_BINARY_DIR}/cmake/CPM_${CPM_DOWNLOAD_VERSION}.cmake")
endif ()
if (NOT (EXISTS ${CPM_DOWNLOAD_LOCATION}))
    message(STATUS "Downloading CPM.cmake to ${CPM_DOWNLOAD_LOCATION}")
    file(DOWNLOAD
            https://github.com/TheLartians/CPM.cmake/releases/download/v${CPM_DOWNLOAD_VERSION}/CPM.cmake
            ${CPM_DOWNLOAD_LOCATION}
            )
endif ()
include(${CPM_DOWNLOAD_LOCATION})


add_subdirectory(highway-experiments)
add_subdirectory(floating-points)
add_subdirectory(compilation-define-test)
add_subdirectory(sys-calls)
add_subdirectory(cache-latency)
add_subdirectory(eigen-experiments)
add_subdirectory(class-experiments)