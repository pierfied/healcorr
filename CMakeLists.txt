cmake_minimum_required(VERSION 3.14)
project(healcorr CUDA CXX)

set(CMAKE_CUDA_STANDARD 11)

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -Xcompiler -fopenmp")
endif()

add_library(healcorr SHARED src/healcorr.cu src/healcorr.h)