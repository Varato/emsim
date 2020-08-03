#include <cuda_runtime.h>
#include <cufft.h>
#include <stdio.h>
#include <math.h>


#include "row_reduce_sum.cuh"
#include "broadcast_mul.cuh"


extern "C" void build_slices_cufft_kernel(float scattering_factors[], int n_elems,
                                          float atom_histograms[], int n_slices, int n1, int n2,
                                          float output[])
/*
    Logical dimensions of the input arrays:
        scattering_factors: (n_elems, n1, n2 // 2 + 1)
        atom_histograms:    (n_elems, n_slices, n1, n2)
    Notice the scattering_factors are halved on theiry last dimension, because it will be used in c2r FFT transforms.
*/

{
    int n[2] = {n1, n2};
    int n2_half = n2 / 2 + 1;
    int n_pix = n1 * n2;
    int n_pix_half = n1 * n2_half;

    int batch = n_elems * n_slices;

    cufftReal* batch_data_d;          // to hold the atom_histograms and the resulted slices in device memory
    cufftReal* scattering_factors_d;  // to hold the halved scattering_factors for n_elems elements
    cufftComplex* location_phase_d;   // to hold intermediate fft result and do computations on it
    if (cudaMalloc((void **)&batch_data_d, sizeof(cufftReal) * batch * n_pix) != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s", cudaGetErrorString(cudaGetLastError()));
    }
    if (cudaMalloc((void **)&scattering_factors_d, sizeof(cufftReal) * n_elems * n_pix_half) != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s", cudaGetErrorString(cudaGetLastError()));
    }
    if (cudaMalloc((void **)&location_phase_d, sizeof(cufftComplex) * batch * n_pix_half) != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s", cudaGetErrorString(cudaGetLastError()));
    }

    if (cudaMemcpy(batch_data_d, atom_histograms, sizeof(float) * batch * n_pix, cudaMemcpyHostToDevice) != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s", cudaGetErrorString(cudaGetLastError()));
    }
    if (cudaMemcpy(scattering_factors_d, scattering_factors, sizeof(float) * n_elems * n_pix_half, cudaMemcpyHostToDevice) != cudaSuccess) {
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
                      CUFFT_R2C, batch) != CUFFT_SUCCESS) {
        fprintf(stderr, "CUFFT error: Plan creation failed");
    }

    if (cufftPlanMany(&ip, 2, n,
                      NULL, 1, n_pix_half,
                      NULL, 1, n_pix,
                      CUFFT_C2R, n_slices) != CUFFT_SUCCESS) {
        fprintf(stderr, "CUFFT error: Plan creation failed");
    }

    if (cufftExecR2C(p, (cufftReal *)batch_data_d, location_phase_d) != CUFFT_SUCCESS) {
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

    if (cufftExecC2R(ip, location_phase_d, batch_data_d) != CUFFT_SUCCESS) {
        fprintf(stderr, "CUFFT error: C2R plan executation failed");
    }


    if (cudaMemcpy(output, batch_data_d, sizeof(float)*n_slices*n_pix, cudaMemcpyDeviceToHost) != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s", cudaGetErrorString(cudaGetLastError()));
    }

    cufftDestroy(p);
    cufftDestroy(ip);
    cudaFree(batch_data_d);
    cudaFree(location_phase_d);
    cudaFree(scattering_factors_d);
}
