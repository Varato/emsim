#ifndef EM_KERNEL_H_
#define EM_KERNEL_H_

#include <fftw3/fftw3.h>

int multislice_propagate_fftw(fftwf_complex wave_in[], int len_x, int len_y, 
                              float slices[], int n_slices,  float pixel_size, float dz,
                              float wave_length, float relativity_gamma,
                              fftwf_complex wave_out[]);

#endif