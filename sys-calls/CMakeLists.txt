cmake_minimum_required(VERSION 3.16)
project(cpp_learning)

set(CMAKE_CXX_STANDARD 17)
find_package(Threads REQUIRED)

add_executable(clone clone.cpp)
target_link_libraries(clone PRIVATE Threads::Threads)
