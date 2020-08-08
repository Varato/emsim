#include <fftw3.h>
#include <omp.h>
#include <stdio.h>

int bin_atoms(float atom_list, float dx, float dy, float dz, int n0, int n1, int n2) {

}


int build_slices_fftwf_kernel(float scattering_factors[], int n_elems,
                              float atom_histograms[], int n_slices, int n1, int n2,
                              float output[])
/*
    Logical dimensions of the input arrays:
        scattering_factors: (n_elems, n1, n2//2 + 1)
        atom_histograms:    (n_elems, n_slices, n1, n2)
    Notice the scattering_factors are halved on theiry last dimension, because it will be used in c2r FFT transforms.
*/
{
    int n_pix = n1 * n2;                   // distance between each transform for the input
    int len_y_half = n2 / 2 + 1;              // the last dimension is halved for dft_r2c
    int n_pix_half = n1 * len_y_half;
    int n[2] = {n1, n2};                   // logical dimensions of each fft transform
    
    fftwf_plan p, ip;
    fftwf_complex *data;

    data = (fftwf_complex *) fftwf_malloc(sizeof(fftwf_complex) * n_elems * n_slices * n_pix_half);

    if(!data){
        return 0;
    }

    if (!fftwf_init_threads()) {
        printf("fftw cannot work with multi threads");
    }

    fftwf_plan_with_nthreads(omp_get_max_threads());
    p  = fftwf_plan_many_dft_r2c(2, n, n_elems * n_slices,
                                 atom_histograms, NULL,
                                 1, n_pix,
                                 data, NULL,
                                 1, n_pix_half,
                                 FFTW_ESTIMATE);

    ip = fftwf_plan_many_dft_c2r(2, n, n_slices,
                                 data, NULL,
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
            float slices_fourier_real = 0;
            float slices_fourier_imag = 0;

            for (int k = 0; k < n_elems; ++k){
                float phase_real = data[k*n_slices*n_pix_half + s*n_pix_half + ii][0];
                float phase_imag = data[k*n_slices*n_pix_half + s*n_pix_half + ii][1];
                float scat_fac = scattering_factors[k*n_pix_half + ii];
                slices_fourier_real += phase_real * scat_fac;
                slices_fourier_imag += phase_imag * scat_fac;
            }
            slices_fourier_real /= (float)n_pix;
            slices_fourier_imag /= (float)n_pix;
            data[s*n_pix_half + ii][0] = slices_fourier_real;
            data[s*n_pix_half + ii][1] = slices_fourier_imag;
        }
    }
    fftwf_execute(ip);
    fftwf_destroy_plan(p);
    fftwf_destroy_plan(ip);
    fftwf_free(data);

    return 1;
}