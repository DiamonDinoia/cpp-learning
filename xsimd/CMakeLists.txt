
CPMAddPackage("gh:xtensor-stack/xsimd#13.2.0")
CPMAddPackage("gh:martinus/nanobench@4.3.11")

add_compile_options(-march=native)
set(CMAKE_CXX_STANDARD 20)

add_executable(reduce_add reduce_add.cpp)
target_link_libraries(reduce_add PRIVATE xsimd nanobench::nanobench)


add_executable(complex_mul complex_multiply.cpp)
target_link_libraries(complex_mul PRIVATE xsimd nanobench::nanobench)