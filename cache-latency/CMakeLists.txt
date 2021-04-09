cmake_minimum_required(VERSION 3.16)
project(cache)

set(CMAKE_C_STANDARD 11)

#set(CMAKE_C_COMPILER gcc-10)

add_compile_options(-O0)

add_executable(cache_test cache_test.c)
