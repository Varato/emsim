cmake_minimum_required(VERSION 3.17)

find_package(Python COMPONENTS Interpreter Development NumPy REQUIRED)
find_package(CUDAToolkit REQUIRED)

add_library(libSliceCuda STATIC
    src/SliceBuilder.cu
    src/SliceBuilder.h
    src/SliceBuilder_kernel.cu
    src/SliceBuilder_kernel.h
    src/utils.cu)

target_link_libraries(libSliceCuda PRIVATE CUDA::cufft)
set_target_properties(libSliceCuda PROPERTIES
        CUDA_ARCHITECTURES 75
        CUDA_SEPARABLE_COMPILATION ON)

pybind11_add_module(dens_kernel_cuda src/dens_kernel_cuda.cpp)
target_link_libraries(dens_kernel_cuda PRIVATE libSliceCuda)

install(TARGETS dens_kernel_cuda
        LIBRARY
        DESTINATION ${CMAKE_SOURCE_DIR}/emsim/ext_cuda)