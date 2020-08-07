cmake_minimum_required(VERSION 3.12)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")

find_package(Python COMPONENTS Interpreter Development NumPy REQUIRED)

if(WIN32)
    message("On Win32 system: using shipped libfftw3f-3.lib and libfftw3f-3.dll")
    set(FFTW_ROOT ${CMAKE_CURRENT_SOURCE_DIR})
endif(WIN32)

find_package(FFTW REQUIRED)
find_package(OpenMP)

if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

message("Found Python include at ${Python_INCLUDE_DIRS}")
message("Found Python libraries: ${Python_LIBRARIES}")
message("Found NumPy include dir: ${Python_NumPy_INCLUDE_DIRS}")
message("Found Python runtime libraries at: ${Python_RUNTIME_LIBRARY_DIRS}")
message("Found FFTW include at: ${FFTW_INCLUDE_DIRS}")
message("Found FFTW float lib at: ${FFTW_FLOAT_LIB}")
message("Found FFTW float threads lib at: ${FFTW_FLOAT_THREADS_LIB}")
message("OpenMP cflags: ${OpenMP_C_FLAGS}")


add_library(dens_kernel MODULE
    src/dens_kernel.c
    src/dens_kernel_pymodule.c)

add_library(em_kernel MODULE
    src/em_kernel.c
    src/em_kernel_pymodule.c)

target_include_directories(dens_kernel PRIVATE 
    ${Python_NumPy_INCLUDE_DIRS}
    ${Python_INCLUDE_DIRS})
target_link_libraries(dens_kernel PRIVATE ${Python_LIBRARIES})

target_include_directories(em_kernel PRIVATE 
    ${Python_NumPy_INCLUDE_DIRS}
    ${Python_INCLUDE_DIRS})
target_link_libraries(em_kernel PRIVATE ${Python_LIBRARIES})

if(WIN32)
    # Assume fftw3 is compiled with --with-combined-threads option on Windows systems,
    # so the Float library contains multi-thread symbols
    if( NOT FFTW_FLOAT_LIB_FOUND )
        message(FATAL_ERROR "libfftw3f not found. Try to provide FFTW_ROOT to cmake")
    endif()
    target_link_libraries(dens_kernel PRIVATE FFTW::Float)
    target_link_libraries(em_kernel PRIVATE FFTW::Float)

    set_target_properties(dens_kernel PROPERTIES SUFFIX ".pyd")
    set_target_properties(em_kernel PROPERTIES SUFFIX ".pyd")

    # copy dll to the pyd directory
    add_custom_command(TARGET dens_kernel POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CMAKE_CURRENT_SOURCE_DIR}/bin/libfftw3f-3.dll" $<TARGET_FILE_DIR:dens_kernel>)
    add_custom_command(TARGET em_kernel POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CMAKE_CURRENT_SOURCE_DIR}/bin/libfftw3f-3.dll" $<TARGET_FILE_DIR:em_kernel>)

else()
    if( NOT FFTW_FLOAT_THREADS_LIB_FOUND )
        message(FATAL_ERROR "libfftw3f_threads not found. Try to provide FFTW_ROOT to cmake")
    endif()

    target_link_libraries(dens_kernel PRIVATE FFTW::FloatThreads)
    target_link_libraries(em_kernel PRIVATE FFTW::FloatThreads)
endif(WIN32)


# if use scikit-build
if(SKBUILD)
    find_package(PythonExtensions REQUIRED)
    python_extension_module(dens_kernel_cuda)
    python_extension_module(em_kernel_cuda)
endif()


install(TARGETS dens_kernel em_kernel
    LIBRARY
    DESTINATION "emsim/ext")

if (WIN32)
    install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/bin/libfftw3f-3.dll"
        DESTINATION "emsim/ext")
endif()