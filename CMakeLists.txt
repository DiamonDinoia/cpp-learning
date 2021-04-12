cmake_minimum_required(VERSION 3.16)

project(cpp_learning)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_VERBOSE_MAKEFILE ON)
set(DCMAKE_RULE_MESSAGES OFF)

set(CMAKE_CXX_STANDARD 17)

add_subdirectory(floating-points)
add_subdirectory(compilation-define-test)
add_subdirectory(sys-calls)
add_subdirectory(cache-latency)