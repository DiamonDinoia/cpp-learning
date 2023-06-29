cmake_minimum_required(VERSION 3.16)


if (CMAKE_BUILD_TYPE MATCHES Debug)
    add_compile_options(-fstack-protector-all)
    add_compile_options(-Wall -Wextra -pedantic)
endif ()
if (CMAKE_BUILD_TYPE MATCHES Release)
    add_compile_options(-ffp-contract=fast )
    add_compile_options(-freciprocal-math -fmerge-all-constants -fno-math-errno)
    # add_compile_options(-march=native)
endif ()


CPMAddPackage(
        NAME Highway
        GIT_REPOSITORY https://github.com/google/highway.git
        GIT_TAG ${HIGHWAY_DOWNLOAD_VERSION}
        GIT_SHALLOW YES
        GIT_PROGRESS YES
        EXCLUDE_FROM_ALL YES
        SYSTEM YES
        OPTIONS
            "HWY_FORCE_STATIC_LIBS Yes"
)

CPMAddPackage(
  NAME Benchmark
  GIT_REPOSITORY https://github.com/google/benchmark.git
  VERSION 1.7.1
  GIT_SHALLOW YES
  GIT_PROGRESS YES
  EXCLUDE_FROM_ALL YES
  SYSTEM YES
  OPTIONS
   "BENCHMARK_ENABLE_TESTING Off"
   "BENCHMARK_USE_LIBCXX On"
)


add_executable(highway highway.cpp)
target_link_libraries(highway PRIVATE hwy benchmark::benchmark)
target_include_directories(highway PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

