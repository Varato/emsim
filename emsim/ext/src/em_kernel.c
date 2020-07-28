#include <fftw3/fftw3.h>
#include <omp.h>
#include <math.h>
#include <stdio.h>


int multislice_propagate_fftw(fftwf_complex wave_in[], int len_x, int len_y, 
                              float slices[], int n_slices,  float pixel_size, float dz,
                              float wave_length, float relativity_gamma,
                              fftwf_complex wave_out[]) {

    printf("begin\n");
    const float PI = 3.14159265359;

    int n_pix = len_x * len_y;
    // frequency resolution
    float dfx = 1.0f/pixel_size/(float)len_x;
    float dfy = 1.0f/pixel_size/(float)len_y;
    //Nyquist frequency and 1/3 filter
    float f_max = 0.5f/pixel_size;
    float filter = 0.6667 * f_max;

    fftwf_plan p, ip;

    fftwf_plan_with_nthreads(omp_get_max_threads());
    p = fftwf_plan_dft_2d(len_x, len_y, wave_out, wave_out, FFTW_FORWARD, FFTW_ESTIMATE);
    ip = fftwf_plan_dft_2d(len_x, len_y, wave_out, wave_out, FFTW_BACKWARD, FFTW_ESTIMATE);


    int ii;  // indexing pixels
    for (ii = 0; ii < n_pix; ++ii) {
    // printf("wave_in %d = (%f, %f)\n", ii, wave_in[ii][0], wave_in[ii][1]);
        wave_out[ii][0] = wave_in[ii][0];
        wave_out[ii][1] = wave_in[ii][1];
    }

    for (int s = 0; s < n_slices; ++s) {

        // multiply wave function by transmission function in real space
        #pragma omp parallel for
        for (ii = 0; ii < n_pix; ++ii){
            float pix = slices[s*n_pix + ii];
            float t_real = cosf(pix * wave_length * relativity_gamma);
            float t_imag = sinf(pix * wave_length * relativity_gamma);
            // (wave_out[ii][0] + wave_out[ii][1] i) * (t_real + t_imag i)
            float real = wave_out[ii][0] * t_real - wave_out[ii][1] * t_imag;
            float imag = wave_out[ii][0] * t_imag + wave_out[ii][1] * t_real;
            wave_out[ii][0] = real;
            wave_out[ii][1] = imag; 
        }

        fftwf_execute(p);

        // in Fourier space, multiply the wave by spatial propagator.
        #pragma omp parallel for
        for (ii = 0; ii < n_pix; ++ii){
            int i = ii / len_y;
            int j = ii % len_y; 
            // ifftshifted indexes
            int is = i + (i<(len_x+1)/2? len_x/2: -(len_x+1)/2);
            int js = j + (j<(len_y+1)/2? len_y/2: -(len_y+1)/2); 
            float fx = (float)(is - len_x/2) * dfx;
            float fy = (float)(js - len_y/2) * dfy;
            float f2 = fx*fx + fy*fy;

            if (f2 <= filter*filter) {
                // construct the spatial propagator;
                float p_real = cosf(wave_length * PI * dz * f2);
                float p_imag = -sinf(wave_length * PI * dz * f2);
                float real = (wave_out[ii][0] * p_real - wave_out[ii][1] * p_imag);
                float imag = (wave_out[ii][0] * p_imag + wave_out[ii][1] * p_real);
                wave_out[ii][0] = real / (float)n_pix;
                wave_out[ii][1] = imag / (float)n_pix;
            }
            
            else {
                wave_out[ii][0] = 0;
                wave_out[ii][1] = 0;
            }
        }
        fftwf_execute(ip);
    }

    fftwf_destroy_plan(p);
    fftwf_destroy_plan(ip);
    return 1;
}