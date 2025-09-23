add_library(kernelgen INTERFACE)

target_include_directories(
    kernelgen
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>)

target_compile_features(kernelgen INTERFACE cxx_std_17)

add_executable(kernelgen_dispatch_sample samples/dispatch_sample.cpp)
target_link_libraries(kernelgen_dispatch_sample PRIVATE kernelgen)
set_target_properties(
    kernelgen_dispatch_sample
    PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS OFF)

add_executable(kernelgen_unroll_sample samples/unroll_sample.cpp)
target_link_libraries(kernelgen_unroll_sample PRIVATE kernelgen)
set_target_properties(
    kernelgen_unroll_sample
    PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS OFF)
