cmake_minimum_required(VERSION 3.16)

project(cpp_learning)

set(CMAKE_CXX_STANDARD 17)

if (CMAKE_BUILD_TYPE MATCHES Debug)
    add_compile_options(-fstack-protector-all)
    add_compile_options(-Wall -Wextra -pedantic)
endif ()
if (CMAKE_BUILD_TYPE MATCHES Release)
    add_compile_options(-ffp-contract=fast)
    add_compile_options(-freciprocal-math -fmerge-all-constants -fno-math-errno)
    add_compile_options(-O3 -march=native -funroll-loops -finline-functions -ftree-vectorize)
endif ()


CPMAddPackage(
        NAME Eigen
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen
        GIT_TAG ${EIGEN_DOWNLOAD_VERSION}
        GIT_SHALLOW YES
        GIT_PROGRESS YES
        EXCLUDE_FROM_ALL YES
        SYSTEM YES
        DOWNLOAD_ONLY YES 

)

if(Eigen_ADDED)
  add_library(Eigen INTERFACE IMPORTED)
  target_include_directories(Eigen INTERFACE ${Eigen_SOURCE_DIR})
endif()


add_executable(eigen_tests eigen-experiments.cpp)
target_link_libraries(eigen_tests Eigen)