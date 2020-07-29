#include <fftw3/fftw3.h>
#include <omp.h>
#include <stdio.h>


int build_slices_fftwf_kernel(float scattering_factors_ifftshifted[], int n_elems,
                              float atom_histograms[], int n_slices, int n1, int n2,
                              float output[])
/*
    Logical dimensions of the input arrays:
        scattering_factors_ifftshifted: (n_elems, n1, n2)
        atom_histograms:    (n_elems, n_slices, n1, n2)

    Here it needs the scattering_factors_ifftshifted to be ifftshifted (i.e. the zero-frequency is at corner). */
{
    int n_pix = n1 * n2;                   // distance between each transform for the input
    int len_y_half = n2 / 2 + 1;              // the last dimension is halved for dft_r2c
    int n_pix_half = n1 * len_y_half;
    int n[2] = {n1, n2};                   // logical dimensions of each fft transform
    
    fftwf_plan p, ip;
    fftwf_complex *location_phase;
    fftwf_complex *slices_flourer;

    location_phase = (fftwf_complex *) fftwf_malloc(sizeof(fftwf_complex) * n_elems * n_slices * n_pix_half);
    slices_flourer = (fftwf_complex *) fftwf_malloc(sizeof(fftwf_complex) * n_slices * n_pix_half);

    if(!location_phase || !slices_flourer){
        return 0;
    }

    if (!fftwf_init_threads()) {
        printf("fftw cannot work with multi threads");
    }

    fftwf_plan_with_nthreads(omp_get_max_threads());
    p  = fftwf_plan_many_dft_r2c(2, n, n_elems * n_slices,
                                 atom_histograms, NULL,
                                 1, n_pix,
                                 location_phase, NULL,
                                 1, n_pix_half,
                                 FFTW_ESTIMATE);

    ip = fftwf_plan_many_dft_c2r(2, n, n_slices,
                                 slices_flourer, NULL,
                                 1, n_pix_half,
                                 output, NULL,
                                 1, n_pix,
                                 FFTW_ESTIMATE);

    fftwf_execute(p);

    // Convolve and sum over elements in fourier space
    int s;
    #pragma omp parallel for
    for (s = 0; s < n_slices; ++s) {
        for (int ii = 0; ii < n_pix_half; ++ii) {
            int i = ii / len_y_half;
            int j = ii % len_y_half;
            slices_flourer[s * n_pix_half + ii][0] = 0;
            slices_flourer[s * n_pix_half + ii][1] = 0;  // (n_slices, n_pix)

            for (int k = 0; k < n_elems; ++k){
                float phase_real = location_phase[k*n_slices*n_pix_half + s*n_pix_half + ii][0];
                float phase_imag = location_phase[k*n_slices*n_pix_half + s*n_pix_half + ii][1];
                float scat_fac = scattering_factors_ifftshifted[k*n_pix + i*n2 + j];
                slices_flourer[s * n_pix_half + ii][0] += phase_real * scat_fac;
                slices_flourer[s * n_pix_half + ii][1] += phase_imag * scat_fac;
            }
            slices_flourer[s * n_pix_half + ii][0] /= (float)n_pix;
            slices_flourer[s * n_pix_half + ii][1] /= (float)n_pix;
        }
    }
    fftwf_execute(ip);
    fftwf_destroy_plan(p);
    fftwf_destroy_plan(ip);
    fftwf_free(location_phase);
    fftwf_free(slices_flourer);

    return 1;
}