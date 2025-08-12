CPMAddPackage("gh:DiamonDinoia/xsimd#improve-sse2-swizzle")
CPMAddPackage("gh:martinus/nanobench@4.3.11")

set(CMAKE_CXX_STANDARD 20)
add_executable(swizzles-sse2 swizzle-sse2.cpp)
target_link_libraries(swizzles-sse2 PRIVATE xsimd nanobench::nanobench)

add_compile_options(-march=native -ftree-vectorize -fcx-limited-range)

add_executable(reduce_add reduce_add.cpp)
target_link_libraries(reduce_add PRIVATE xsimd nanobench::nanobench)

add_executable(complex_mul complex_multiply.cpp)
target_link_libraries(complex_mul PRIVATE xsimd nanobench::nanobench)

add_executable(swizzles swizzles.cpp)
target_link_libraries(swizzles PRIVATE xsimd nanobench::nanobench)

