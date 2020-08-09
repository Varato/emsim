cmake_minimum_required(VERSION 3.17)

find_package(Python COMPONENTS Interpreter Development NumPy REQUIRED)
find_package(CUDAToolkit REQUIRED)

add_library(libSliceCuda STATIC
    src/SliceBuilder.cpp
    src/SliceBuilder.h
    src/SliceBuilder_kernel.cu
    src/utils.cu src/SliceBuilder_kernel.h)

target_link_libraries(libSliceCuda PRIVATE CUDA::cufft)