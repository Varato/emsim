#ifndef DENS_KERNEL_CUH_
#define DENS_KERNEL_CUH_

void build_slices_fourier_cuda_device(float scattering_factors_d[], int n_elems,
                                      float atom_histograms_d[], int n_slices, int n1, int n2,
                                      float output_d[]);

#endif