cmake_minimum_required(VERSION 3.14)
project(healcorr CUDA CXX)

set(CMAKE_CUDA_STANDARD 11)

find_package(OpenMP)
IF (OPENMP_FOUND)
    SET(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=${OpenMP_CXX_FLAGS}")
    SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_CXX_FLAGS}")
ENDIF()

SET(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -arch=sm_75")

add_library(healcorr SHARED src/healcorr.cu src/healcorr.h)