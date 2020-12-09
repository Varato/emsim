cmake_minimum_required(VERSION 3.17)


enable_language(CUDA)

find_package(CUDAToolkit REQUIRED)


add_library(libSliceCuda STATIC
    src/SliceBuilder.cu
    src/SliceBuilder_kernel.cu
    src/utils.cu)

target_link_libraries(libSliceCuda PRIVATE CUDA::cufft)
set_target_properties(libSliceCuda PROPERTIES
        CUDA_ARCHITECTURES 61
        CUDA_SEPARABLE_COMPILATION ON)


add_library(libWaveCuda STATIC
        src/WavePropagator_kernel.cu
        src/WavePropagator.cu)
target_link_libraries(libWaveCuda PRIVATE CUDA::cufft)
set_target_properties(libWaveCuda PROPERTIES
        CUDA_ARCHITECTURES 61
        CUDA_SEPARABLE_COMPILATION ON)


# --- dens_kernel_cuda ---
pybind11_add_module(dens_kernel_cuda src/dens_kernel_cuda.cpp)
target_link_libraries(dens_kernel_cuda PRIVATE libSliceCuda)
set_target_properties(dens_kernel_cuda PROPERTIES
        CUDA_ARCHITECTURES 61
        CUDA_SEPARABLE_COMPILATION ON)


# --- wave_kernel_cuda ---
pybind11_add_module(wave_kernel_cuda src/wave_kernel_cuda.cpp)
target_link_libraries(wave_kernel_cuda PRIVATE libWaveCuda)
set_target_properties(wave_kernel_cuda PROPERTIES
        CUDA_ARCHITECTURES 61
        CUDA_SEPARABLE_COMPILATION ON)
