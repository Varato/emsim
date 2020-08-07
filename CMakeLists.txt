cmake_minimum_required(VERSION 3.17)
# since VERSION 3.10, CUDA becomes an intrinsic language to cmake
# since VERSION 3.12, can use FindPython
# since VERSION 3.17, can use FindCUDAToolkit

project(emsim C CXX CUDA)

add_subdirectory("emsim/ext")
# add_subdirectory("emsim/ext_cuda")

