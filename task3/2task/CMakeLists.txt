cmake_minimum_required(VERSION 3.16)
project(task3_2 CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(task3_2 2.cpp)
target_compile_options(task3_2 PRIVATE -O2)
target_link_libraries(task3_2 PRIVATE pthread)
