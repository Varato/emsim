find_package(Python COMPONENTS Interpreter Development NumPy REQUIRED)

message("Found Python libs at ${Python_LIBRARY_DIRS}")
message("Found NumPy include dir ${Python_NumPy_INCLUDE_DIRS}")

add_library(libdensKernelCuda src/dens_kernel.cu)
set_property(TARGET libdensKernelCuda PROPERTY CUDA_ARCHITECTURES 75)
