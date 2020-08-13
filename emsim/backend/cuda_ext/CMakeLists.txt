cmake_minimum_required(VERSION 3.17)

#find_package(Python COMPONENTS Interpreter Development NumPy REQUIRED)
find_package(CUDAToolkit REQUIRED)


add_library(libSliceCuda STATIC
    src/SliceBuilder.cu
    src/SliceBuilder_kernel.cu
    src/utils.cu)

target_link_libraries(libSliceCuda PRIVATE CUDA::cufft)
set_target_properties(libSliceCuda PROPERTIES
        CUDA_ARCHITECTURES 75
        CUDA_SEPARABLE_COMPILATION ON)


add_library(libWaveCuda STATIC
        src/WavePropagator_kernel.cu
        src/WavePropagator.cu)
target_link_libraries(libWaveCuda PRIVATE CUDA::cufft)
set_target_properties(libWaveCuda PROPERTIES
        CUDA_ARCHITECTURES 75
        CUDA_SEPARABLE_COMPILATION ON)


# --- dens_kernel_cuda ---
pybind11_add_module(dens_kernel_cuda src/dens_kernel_cuda.cpp)
target_link_libraries(dens_kernel_cuda PRIVATE libSliceCuda)


# --- em_kernel_cuda ---
pybind11_add_module(em_kernel_cuda src/em_kernel_cuda.cpp)
target_link_libraries(em_kernel_cuda PRIVATE libWaveCuda)


if(WIN32)
#    set_target_properties(dens_kernel_cuda PROPERTIES SUFFIX ".pyd")
#    set_target_properties(em_kernel_cuda PROPERTIES SUFFIX ".pyd")

    # copy dll to the pyd directory
    file(GLOB CUDART_DLL "${CUDAToolkit_BIN_DIR}/cudart64*.dll")
    file(GLOB CUFFT_DLL "${CUDAToolkit_BIN_DIR}/cufft64*.dll")
    message("find cudart64*.dll: ${CUDART_DLL}")
    message("find cufft64*.dll: ${CUFFT_DLL}")

    add_custom_command(TARGET dens_kernel_cuda POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CUDART_DLL} $<TARGET_FILE_DIR:dens_kernel_cuda>)
    add_custom_command(TARGET dens_kernel_cuda POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CUFFT_DLL} $<TARGET_FILE_DIR:dens_kernel_cuda>)

    add_custom_command(TARGET em_kernel_cuda POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CUDART_DLL} $<TARGET_FILE_DIR:em_kernel_cuda>)
    add_custom_command(TARGET em_kernel_cuda POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CUFFT_DLL} $<TARGET_FILE_DIR:em_kernel_cuda>)


    # copy dll to the pyd directory
    file(GLOB CUDART_DLL "${CUDAToolkit_BIN_DIR}/cudart64*.dll")
    file(GLOB CUFFT_DLL "${CUDAToolkit_BIN_DIR}/cufft64*.dll")
    install(FILES ${CUDART_DLL} ${CUFFT_DLL}
            DESTINATION "emsim/ext_cuda")
endif()