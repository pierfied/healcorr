cmake_minimum_required(VERSION 3.14)
project(healcorr CUDA CXX)

set(CMAKE_CUDA_STANDARD 11)

add_library(healcorr SHARED src/healcorr.cu src/healcorr.h)