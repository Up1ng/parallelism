cmake_minimum_required(VERSION 3.16)
project(Task2_2 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_executable(task2_integrate_omp task2_integrate_omp.cpp)

find_package(OpenMP REQUIRED)
target_link_libraries(task2_integrate_omp PRIVATE OpenMP::OpenMP_CXX)
