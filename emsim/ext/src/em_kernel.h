#ifndef EM_KERNEL_H_
#define EM_KERNEL_H_

typedef float fftwf_complex[2];

int multislice_propagate_fftw(fftwf_complex wave_in[], int n1, int n2, 
                              float slices[], int n_slices,  float pixel_size, float dz,
                              float wave_length, float relativity_gamma,
                              fftwf_complex wave_out[]);

int lens_propagate_fftw(fftwf_complex wave_in[], int n1, int n2, float pixel_size,
                        float wave_length, float cs_mm, float defocus, float aperture,
                        fftwf_complex wave_out[]);

#endif