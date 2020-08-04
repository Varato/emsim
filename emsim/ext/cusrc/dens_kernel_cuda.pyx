import cupy as cp
import numpy as np

cdef extern from "dens_kernel_cuda.h":
    void build_slices_fourier_cuda_device(float scattering_factors_d[], int n_elems,
                                          float atom_histograms_d[], int n_slices, int n1, int n2,
                                          float output_d[])


cdef build_slices_fourier_cuda_wrapper(size_t scattering_factors_ptr, int n_elems,
                                       size_t atom_histograms_ptr, int n_slices, int n1, int n2,
                                       size_t output_ptr):
    build_slices_fourier_cuda_device(
        <float*>scattering_factors_ptr, n_elems,
        <float*>atom_histograms_ptr, n_slices, n1, n2,
        <float*>output_ptr
    )


def build_slices_fourier_cuda(scattering_factors, atom_histograms):
    n_elems = atom_histograms.shape[0]
    n_slices = atom_histograms.shape[1]
    n1, n2 = atom_histograms.shape[2:]

    output = cp.empty(shape=(n_slices, n1, n2), dtype=np.float32)

    build_slices_fourier_cuda_wrapper(
        scattering_factors.data.ptr, n_elems, 
        atom_histograms.data.ptr, n_slices, n1, n2,
        output.data.ptr)
    return output
