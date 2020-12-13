cmake_minimum_required(VERSION 3.17)
# since VERSION 3.10, CUDA becomes an intrinsic language to cmake
# since VERSION 3.12, can use FindPython
# since VERSION 3.17, can use FindCUDAToolkit

project(emsim C CXX)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(WITH_FFTW_EXT FALSE CACHE BOOL "build with fftw extension")
set(WITH_CUDA_EXT FALSE CACHE BOOL "build with fftw extension")

find_package(CUDAToolkit QUIET)

add_subdirectory(pybind11)

if ( WITH_FFTW_EXT )
    message("-- Will build with fftw extensions")
    add_subdirectory("emsim/backend/fftw_ext")
endif()

if ( CUDAToolkit_FOUND AND WITH_CUDA_EXT )
    message("-- Will build with cuda extensions because CUDA is found.")
    add_subdirectory(emsim/backend/cuda_ext)
endif()

