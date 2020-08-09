#include <cuda_runtime.h>
//#include <thrust/device_vector.h>
#include <cufft.h>
#include <stdio.h>
#include <math.h>


#include "utils.h"




void build_slices_fourier_cuda_device(float scattering_factors_d[], int n_elems,
                                      float atom_histograms_d[], int n_slices, int n1, int n2,
                                      float output_d[])
/*
 * Logical dimensions of the arrays:
 *     scattering_factors_d: (n_elems, n1, n2 // 2 + 1)
 *     atom_histograms_d:    (n_elems, n_slices, n1, n2)
 *     output_d            : (n_slices, n1, n2)
 * They must be in device memory.
 * 
 * Notice the scattering_factors are halved on their last dimension, because it will be used in c2r FFT transforms.
 */

{
    int n[2] = {n1, n2};
    int n2_half = n2 / 2 + 1;
    int n_pix = n1 * n2;
    int n_pix_half = n1 * n2_half;

    //TODO use thurst vector here
    cufftComplex* location_phase_d;   // to hold intermediate fft result and do computations on it

    if (cudaMalloc((void **)&location_phase_d, sizeof(cufftComplex) * n_elems * n_slices * n_pix_half) != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s", cudaGetErrorString(cudaGetLastError()));
    }
    
    cufftHandle p, ip;
    /*
     * cufftResult cufftPlanMany(cufftHandle *plan, int rank, int *n, 
     *                           int *inembed, int istride, int idist, 
     *                           int *onembed, int ostride, int odist, 
     *                           cufftType type, int batch);
     */
    if (cufftPlanMany(&p, 2, n,
                      NULL, 1, n_pix, 
                      NULL, 1, n_pix_half,
                      CUFFT_R2C, n_elems * n_slices) != CUFFT_SUCCESS) {
        fprintf(stderr, "CUFFT error: Plan creation failed");
    }

    if (cufftPlanMany(&ip, 2, n,
                      NULL, 1, n_pix_half,
                      NULL, 1, n_pix,
                      CUFFT_C2R, n_slices) != CUFFT_SUCCESS) {
        fprintf(stderr, "CUFFT error: Plan creation failed");
    }

    if (cufftExecR2C(p, (cufftReal *)atom_histograms_d, location_phase_d) != CUFFT_SUCCESS) {
        fprintf(stderr, "CUFFT error: R2C plan executation failed");
    }

    broadCastMul(location_phase_d, scattering_factors_d, 1.0f/(float)n_pix, n_elems, n_slices, n_pix_half);
    if (cudaGetLastError() != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s", cudaGetErrorString(cudaGetLastError()));
    }
    rowReduceSum(location_phase_d, n_elems, n_slices*n_pix_half, location_phase_d);
    if (cudaGetLastError() != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s", cudaGetErrorString(cudaGetLastError()));
    }

    if (cufftExecC2R(ip, location_phase_d, output_d) != CUFFT_SUCCESS) {
        fprintf(stderr, "CUFFT error: C2R plan executation failed");
    }

    cufftDestroy(p);
    cufftDestroy(ip);
    cudaFree(location_phase_d);
}
