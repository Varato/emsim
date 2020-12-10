cmake_minimum_required(VERSION 3.12)


set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")

find_package(Python COMPONENTS Interpreter Development NumPy REQUIRED)

if(WIN32)
    set(FFTW_ROOT ${CMAKE_CURRENT_SOURCE_DIR})
    message("-- On Win32 system: using shipped libfftw3f-3.lib and libfftw3f-3.dll")
    message("-- FFTW_ROOT = ${FFTW_ROOT}")
endif(WIN32)
find_package(FFTW REQUIRED)

find_package(OpenMP)
if (OPENMP_FOUND)
    message("-- Found OpenMP library ${OpenMP_CXX_LIBRARIES}")
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()


add_library(libSlice STATIC
    src/SliceBuilder.cpp
    src/SliceBuilder.h src/common.h)


add_library(libWave STATIC
    src/WavePropagator.cpp
    src/WavePropagator.h)

target_link_libraries(libSlice PRIVATE OpenMP::OpenMP_CXX)
target_link_libraries(libWave PRIVATE OpenMP::OpenMP_CXX)

if(WIN32)
    # Assume fftw3 is compiled with --with-combined-threads option on Windows systems,
    # so the Float library contains multi-thread symbols
    if( NOT FFTW_FLOAT_LIB_FOUND )
        message(FATAL_ERROR "libfftw3f not found. Try to provide FFTW_ROOT to cmake")
    endif()
    target_link_libraries(libSlice PUBLIC FFTW::Float)
    target_link_libraries(libWave PUBLIC FFTW::Float)

else()
    if( NOT FFTW_FLOAT_THREADS_LIB_FOUND )
        message(FATAL_ERROR "libfftw3f_threads not found. Try to provide FFTW_ROOT to cmake")
    endif()
    target_link_libraries(libSlice PUBLIC FFTW::FloatThreads)
    target_link_libraries(libWave PUBLIC FFTW::FloatThreads)
endif(WIN32)

pybind11_add_module(slice_kernel src/slice_kernel.cpp)
target_link_libraries(slice_kernel PRIVATE libSlice)

pybind11_add_module(wave_kernel src/wave_kernel.cpp)
target_link_libraries(wave_kernel PRIVATE libWave)
