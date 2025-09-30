CPMAddPackage("gh:martinus/nanobench@4.3.11")

set(CMAKE_CXX_STANDARD 17)
add_compile_options(-ftree-vectorize -fcx-limited-range)

add_executable(dispatch-benchmark dispatch-benchmark.cpp)
target_link_libraries(dispatch-benchmark PRIVATE nanobench::nanobench)

add_executable(dispatch-test dispatch-test.cpp)
target_link_libraries(dispatch-test PRIVATE nanobench::nanobench)
