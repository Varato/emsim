#include <fftw3/fftw3.h>
#include <omp.h>

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
    int n[2] = {len_x, len_y};                   // logical dimensions of each fft transform
    int n_pix = len_x * len_y;  // distance between each transform for the input
    
    fftwf_plan p, ip;
    fftwf_complex *in, *location_phase;
    fftwf_complex *slices_flourer;

    in  = (fftwf_complex *) fftwf_malloc(sizeof(fftwf_complex) * n_elems * n_slices * n_pix);
    location_phase = (fftwf_complex *) fftwf_malloc(sizeof(fftwf_complex) * n_elems * n_slices * n_pix);
    slices_flourer = (fftwf_complex *) fftwf_malloc(sizeof(fftwf_complex) * n_slices * n_pix);

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
     */
    p  = fftwf_plan_many_dft(2, n, n_elems * n_slices,
                            in, n,
                            1, n_pix,
                            location_phase, n,
                            1, n_pix,
                            FFTW_FORWARD, FFTW_ESTIMATE);
    ip = fftwf_plan_many_dft(2, n, n_slices,
                            slices_flourer, n,
                            1, n_pix,
                            in, n,
                            1, n_pix,
                            FFTW_BACKWARD, FFTW_ESTIMATE);

    //TODO: omit this copy
    #pragma omp parallel for
    for (int I = 0; I < n_elems * n_slices * n_pix; ++I){
        // int t = I / (n_pix * n_slices);
        // int II = I % (n_pix * n_slices);
        in[I][0] = (float)atom_histograms[I];
        in[I][1] = 0;
    }

    fftwf_execute(p);

    // Convolve and sum over elements in fourier space
    for (int s = 0; s < n_slices; ++s) {
        for (int i = 0; i < n_pix; ++i) {
            slices_flourer[s * n_pix + i][0] = 0;
            slices_flourer[s * n_pix + i][1] = 0;  // (n_slices, n_pix)

            for (int k = 0; k < n_elems; ++k){    
                slices_flourer[s * n_pix + i][0] +=
                    location_phase[k*n_slices*n_pix + s*n_pix + i][0] * (float)scattering_factors_ifftshifted[k*n_pix + i];
                slices_flourer[s * n_pix + i][1] +=
                    location_phase[k*n_slices*n_pix + s*n_pix + i][1] * (float)scattering_factors_ifftshifted[k*n_pix + i];
            }
        }
    }

    fftwf_execute(ip);

    for (int i = 0; i < n_slices * n_pix; ++i) {
        output[i] = (float)in[i][0] / (float)n_pix;
    }

    fftwf_destroy_plan(p);
    fftwf_destroy_plan(ip);
    fftwf_free(in);
    fftwf_free(location_phase);
    fftwf_free(slices_flourer);

    return 1;
}