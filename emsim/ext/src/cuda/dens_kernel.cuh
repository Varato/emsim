#ifndef DENS_KERNEL_CUH_
#define DENS_KERNEL_CUH_

int build_slices_cufft_kernel(float scattering_factors[], int n_elems,
                              float atom_histograms[], int n_slices, int n1, int n2,
                              float output[]);

#endif