import cupy as cp
import numpy as np


cdef extern from "<cufft.h>":
    ctypedef float cufftReal
    struct float2
    ctypedef float2 cufftComplex
    
cdef extern from "em_kernel_cuda.h":
    void multislice_propagate_cuda_device(cufftComplex *waveIn_d, int n1, int n2,
                                          cufftReal *slices_d, int nSlices, float pixSize, float dz,
                                          float waveLength, float relativityGamma,
                                          cufftComplex *waveOut_d)

    void lens_propagate_cuda_device(cufftComplex *waveIn_d, int n1, int n2, float pixSize,
                                    float waveLength, float cs_mm, float defocus, float aperture,
                                    cufftComplex *waveOut_d)


cdef multislice_propagate_cufft_device_wrapper(size_t waveIn_d_ptr, int n1, int n2,
                                               size_t slices_d_ptr, int nSlices, float pixSize, float dz,
                                               float waveLength, float relativityGamma,
                                               size_t waveOut_d_ptr):
    multislice_propagate_cuda_device(
        <cufftComplex *>waveIn_d_ptr, n1, n2,
        <cufftReal *>slices_d_ptr, nSlices, pixSize, dz,
        waveLength, relativityGamma,
        <cufftComplex *>waveOut_d_ptr
    )


cdef lens_propagate_cuda_device_wrapper(size_t waveIn_d_ptr, int n1, int n2, float pixSize,
                                        float waveLength, float cs_mm, float defocus, float aperture,
                                        size_t waveOut_d_ptr):

    lens_propagate_cuda_device(
        <cufftComplex *>waveIn_d_ptr, n1, n2, pixSize,
        waveLength, cs_mm, defocus, aperture,
        <cufftComplex *>waveOut_d_ptr
    )


def multislice_propagate_cuda(wave_in, slices, pixel_size, dz, wave_length, relativity_gamma):
    n_slices = slices.shape[0]
    n1, n2 = slices.shape[1:]
    wave_in_ptr = wave_in.data.ptr
    slices_ptr = slices.data.ptr

    wave_out = cp.empty(shape=(n1, n2), dtype=np.complex64)
    wave_out_ptr = wave_out.data.ptr

    multislice_propagate_cufft_device_wrapper(
        wave_in_ptr, n1, n2,
        slices_ptr, n_slices, pixel_size, dz,
        wave_length, relativity_gamma,
        wave_out_ptr
    )
    return wave_out


def lens_propagate_cuda(wave_in, pixel_size, wave_length, cs_mm, defocus, aperture):
    n1, n2 = wave_in.shape

    wave_in_ptr = wave_in.data.ptr

    wave_out = cp.empty(shape=(n1, n2), dtype=np.complex64)
    wave_out_ptr = wave_out.data.ptr

    lens_propagate_cuda_device_wrapper(
        wave_in_ptr, n1, n2, pixel_size,
        wave_length, cs_mm, defocus, aperture,
        wave_out_ptr
    )

    return wave_out