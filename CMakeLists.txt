cmake_minimum_required(VERSION 3.14)
project(healcorr C)

set(CMAKE_C_STANDARD 11)

add_library(healcorr SHARED src/healcorr.c src/healcorr.h)