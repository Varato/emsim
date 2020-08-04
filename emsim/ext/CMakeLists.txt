find_package(Python COMPONENTS Interpreter Development NumPy REQUIRED)
find_package(CUDAToolkit REQUIRED)

# message("Found Python include at ${Python_INCLUDE_DIRS}")
# message("Found Python Libraries: ${Python_LIBRARIES}")
# message("Found NumPy include dir: ${Python_NumPy_INCLUDE_DIRS}")
# message("Found Python Runtime Libraries at: ${Python_RUNTIME_LIBRARY_DIRS}")
# message("Found CudaToolkit Lib Path: ${CUDAToolkit_LIBRARY_DIR}")

add_library(libDensKernelCuda STATIC 
    cusrc/dens_kernel_cuda.cu
    cusrc/broadcast_mul.cu
    cusrc/row_reduce_sum.cu)

target_link_libraries(libDensKernelCuda PRIVATE CUDA::cufft CUDA::cudart)
set_target_properties(libDensKernelCuda PROPERTIES
    CUDA_ARCHITECTURES 75
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_SEPARABLE_COMPILATION ON)


# add_library(dens_kernel_cuda SHARED src/dens_kernel_cuda_pymodule.c)
# target_link_libraries(dens_kernel_cuda PRIVATE libDensKernelCuda ${Python_LIBRARIES})
# target_include_directories(dens_kernel_cuda PRIVATE
#     ${Python_INCLUDE_DIRS}
#     ${Python_NumPy_INCLUDE_DIRS})
# set_target_properties(dens_kernel_cuda PROPERTIES
#     CUDA_ARCHITECTURES 75
#     CUDA_RESOLVE_DEVICE_SYMBOLS ON
#     CUDA_SEPARABLE_COMPILATION ON
#     SUFFIX ".pyd")

install(TARGETS libDensKernelCuda
    DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/lib_x86_64-win32)