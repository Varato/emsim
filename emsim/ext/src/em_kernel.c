#include <fftw3/fftw3.h>
#include <omp.h>
#include <math.h>

/*
 * all real length quantities are in unit of Angstroms except explicitly specified (e.g. cs_mm is in mm)
 */

int multislice_propagate_fftw(fftwf_complex wave_in[], int n1, int n2,
                              float slices[], int n_slices,  float pixel_size, float dz,
                              float wave_length, float relativity_gamma,
                              fftwf_complex wave_out[]) {
    const float PI = 3.14159265359;

    int n_pix = n1 * n2;
    // frequency resolution
    float dfx = 1.0f/pixel_size/(float)n1;
    float dfy = 1.0f/pixel_size/(float)n2;
    //Nyquist frequency and 1/3 filter
    float f_max = 0.5f/pixel_size;
    float filter = 0.6667 * f_max;

    fftwf_plan p, ip;

    fftwf_plan_with_nthreads(omp_get_max_threads());
    p = fftwf_plan_dft_2d(n1, n2, wave_out, wave_out, FFTW_FORWARD, FFTW_ESTIMATE);
    ip = fftwf_plan_dft_2d(n1, n2, wave_out, wave_out, FFTW_BACKWARD, FFTW_ESTIMATE);


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
            int i = ii / n2;
            int j = ii % n2;
            // ifftshifted indexes
            int is = i + (i<(n1+1)/2? n1/2: -(n1+1)/2);
            int js = j + (j<(n2+1)/2? n2/2: -(n2+1)/2);
            float fx = (float)(is - n1/2) * dfx;
            float fy = (float)(js - n2/2) * dfy;
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


//int lens_propagate_fftw(fftwf_complex wave_in[], int n1, int n2,
//                        float wave_length, float cs_mm, float defocus)