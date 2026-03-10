cmake_minimum_required(VERSION 3.16)
project(Task2_3 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenMP REQUIRED)

add_executable(a Var_A.cpp)
add_executable(b Var_B.cpp)

target_link_libraries(a PRIVATE OpenMP::OpenMP_CXX)
target_link_libraries(b PRIVATE OpenMP::OpenMP_CXX)
