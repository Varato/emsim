#ifndef DENS_KERNEL_H_
#define DENS_KERNEL_H_

int build_slices_fftwf_kernel(float scattering_factors_ifftshifted[], int n_elems,
                              float atom_histograms[], int n_slices, int len_x, int len_y,
                              float output[]);

#endif