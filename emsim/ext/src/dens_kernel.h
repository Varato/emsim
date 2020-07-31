#ifndef DENS_KERNEL_H_
#define DENS_KERNEL_H_

int build_slices_fftwf_kernel(float scattering_factors[], int n_elems,
                              float atom_histograms[], int n_slices, int n1, int n2,
                              float output[]);

#endif