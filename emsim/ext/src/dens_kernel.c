#include <fftw3/fftw3.h>
#include <omp.h>
#include <stdio.h>

#include "utils.h"


int build_slices_fftwf_kernel(float scattering_factors_ifftshifted[], int n_elems,
                              float atom_histograms[], int n_slices, int len_x, int len_y,
                              float output[])
    /*
       Logical dimensions of the input arrays:
            scattering_factors_ifftshifted: (n_elems, len_x, len_y)
            atom_histograms:    (n_elems, n_slices, len_x, len_y)

       Here it needs the scattering_factors_ifftshifted to be ifftshifted (i.e. the zero-frequency is at corner).
    
    */
{
    int n_pix = len_x * len_y;                   // distance between each transform for the input
    int len_y_half = len_y / 2 + 1;              // the last dimension is halved for dft_r2c
    int n_pix_half = len_x * len_y_half;
    int n[2] = {len_x, len_y};                   // logical dimensions of each fft transform
    int n_half[2] = {len_x, len_y_half};

    
    fftwf_plan p, ip;
    float *in;
    fftwf_complex *location_phase;
    fftwf_complex *slices_flourer;

    in  = (float *) fftwf_malloc(sizeof(float) * n_elems * n_slices * n_pix);
    location_phase = (fftwf_complex *) fftwf_malloc(sizeof(fftwf_complex) * n_elems * n_slices * n_pix_half);
    slices_flourer = (fftwf_complex *) fftwf_malloc(sizeof(fftwf_complex) * n_slices * n_pix_half);

    if(!in || !location_phase || !slices_flourer){
        return 0;
    }

    fftwf_plan_with_nthreads(omp_get_max_threads());
    /* Signature
       fftwf_plan fftwf_plan_many_dft(int rank, const int *n, int howmany,
                                      fftwf_complex *in, const int *inembed,
                                      int istride, int idist,
                                      fftwf_complex *out, const int *onembed,
                                      int ostride, int odist,
                                      int sign, unsigned flags);

       fftw_plan fftw_plan_many_dft_r2c(int rank, const int *n, int howmany,
                                        double *in, const int *inembed,
                                        int istride, int idist,
                                        fftw_complex *out, const int *onembed,
                                        int ostride, int odist,
                                        unsigned flags);
     */
    p  = fftwf_plan_many_dft_r2c(2, n, n_elems * n_slices,
                                 in, NULL,
                                 1, n_pix,
                                 location_phase, NULL,
                                 1, n_pix_half,
                                 FFTW_ESTIMATE);

    ip = fftwf_plan_many_dft_c2r(2, n, n_slices,
                                 slices_flourer, NULL,
                                 1, n_pix_half,
                                 in, NULL,
                                 1, n_pix,
                                 FFTW_ESTIMATE);

    //TODO: omit this copy
    #pragma omp parallel for
    for (int I = 0; I < n_elems * n_slices * n_pix; ++I){
        in[I] = (float)atom_histograms[I];
    }
    fftwf_execute(p);

    // Convolve and sum over elements in fourier space
    for (int s = 0; s < n_slices; ++s) {
        for (int ii = 0; ii < n_pix_half; ++ii) {
            int i = ii / len_y_half;
            int j = ii % len_y_half;
            slices_flourer[s * n_pix_half + ii][0] = 0;
            slices_flourer[s * n_pix_half + ii][1] = 0;  // (n_slices, n_pix)

            for (int k = 0; k < n_elems; ++k){
                float phase_real = location_phase[k*n_slices*n_pix_half + s*n_pix_half + ii][0];
                float phase_imag = location_phase[k*n_slices*n_pix_half + s*n_pix_half + ii][1];
                float scat_fac = scattering_factors_ifftshifted[k*n_pix + i*len_y + j];
                slices_flourer[s * n_pix_half + ii][0] += phase_real * scat_fac;
                slices_flourer[s * n_pix_half + ii][1] += phase_imag * scat_fac;
            }
        }
    }
    fftwf_execute(ip);

    for (int i = 0; i < n_slices * n_pix; ++i) {
        output[i] = (float)in[i] / (float)n_pix;
    }

    fftwf_destroy_plan(p);
    fftwf_destroy_plan(ip);
    fftwf_free(in);
    fftwf_free(location_phase);
    fftwf_free(slices_flourer);

    return 1;
}