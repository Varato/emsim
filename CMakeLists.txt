cmake_minimum_required(VERSION 3.12)
# since VERSION 3.10, CUDA becomes an intrinsic language to cmake
# since VERSION 3.12, can use FindPython

project(emsim C CXX CUDA)

add_subdirectory("emsim/ext")

