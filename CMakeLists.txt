cmake_minimum_required(VERSION 3.17)
# since VERSION 3.10, CUDA becomes an intrinsic language to cmake
# since VERSION 3.12, can use FindPython
# since VERSION 3.17, can use FindCUDAToolkit

project(emsim C CXX)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(CUDAToolkit QUIET)

add_subdirectory(pybind11)
add_subdirectory("emsim/backend/fftw_ext")

if ( CUDAToolkit_FOUND )
    message("-- Will build with cuda extensions because CUDA is found.")
    add_subdirectory(emsim/backend/cuda_ext)
endif()

