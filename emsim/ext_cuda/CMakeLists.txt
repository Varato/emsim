cmake_minimum_required(VERSION 3.17)

find_package(Python COMPONENTS Interpreter Development NumPy REQUIRED)
find_package(CUDAToolkit REQUIRED)


add_library(libDensKernelCuda STATIC
    src/SliceBuilder.cpp
    src/SliceBuilder.h
    src/SliceBuilder_kernel.cu
    src/utils.cu)
target_link_libraries(libDensKernelCuda PRIVATE CUDA::cufft)
set_target_properties(libDensKernelCuda PROPERTIES
    CUDA_ARCHITECTURES 75
    CUDA_SEPARABLE_COMPILATION ON)


add_library(libEmKernelCuda STATIC
        src/em_kernel_cuda.cu)
target_link_libraries(libEmKernelCuda PRIVATE CUDA::cufft)
set_target_properties(libEmKernelCuda PROPERTIES
    CUDA_ARCHITECTURES 75
    CUDA_SEPARABLE_COMPILATION ON)


# Python extension modules
add_library(dens_kernel_cuda MODULE cython_src/dens_kernel_cuda.cpp)
target_link_libraries(dens_kernel_cuda libDensKernelCuda)
target_include_directories(dens_kernel_cuda PRIVATE
    ${Python_INCLUDE_DIRS}
    ${Python_NumPy_INCLUDE_DIRS})
set_target_properties(dens_kernel_cuda PROPERTIES
    CUDA_ARCHITECTURES 75
    CUDA_SEPARABLE_COMPILATION ON)


add_library(em_kernel_cuda MODULE cython_src/em_kernel_cuda.cpp)
target_include_directories(em_kernel_cuda PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
target_link_libraries(em_kernel_cuda libEmKernelCuda)
target_include_directories(em_kernel_cuda PRIVATE
    ${Python_INCLUDE_DIRS}
    ${Python_NumPy_INCLUDE_DIRS})
set_target_properties(em_kernel_cuda PROPERTIES
    CUDA_ARCHITECTURES 75
    CUDA_SEPARABLE_COMPILATION ON)

if(WIN32)
    set_target_properties(dens_kernel_cuda PROPERTIES SUFFIX ".pyd")
    set_target_properties(em_kernel_cuda PROPERTIES SUFFIX ".pyd")

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
endif()

# if use scikit-build
if(SKBUILD)
    find_package(PythonExtensions REQUIRED)
    # scikit-build will link libpython here
    python_extension_module(dens_kernel_cuda)
    python_extension_module(em_kernel_cuda)
else()
    target_link_libraries(dens_kernel_cuda ${Python_LIBRARIES})
    target_link_libraries(em_kernel_cuda ${Python_LIBRARIES})
endif()

install(TARGETS dens_kernel_cuda em_kernel_cuda
    LIBRARY
    DESTINATION "emsim/ext_cuda")

if (WIN32)
    # copy dll to the pyd directory
    file(GLOB CUDART_DLL "${CUDAToolkit_BIN_DIR}/cudart64*.dll")
    file(GLOB CUFFT_DLL "${CUDAToolkit_BIN_DIR}/cufft64*.dll")
    install(FILES ${CUDART_DLL} ${CUFFT_DLL}
        DESTINATION "emsim/ext_cuda")
endif()