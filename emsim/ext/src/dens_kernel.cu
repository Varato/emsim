#include <cuda_runtime.h>
#include <cufft.h>


__global__ void convolve_fourier(cufftComplex *location_phase, 
                                 cufftReal *scattering_factors,
                                 int n_elems, int n_slices, int n1, int n2)
/*
 * location_phase:     (n_elems, n_slices, n1, n2 // 2 + 1)
 * scattering_factors:           (n_elems, n1, n2 // 2 + 1)
 * k indexes elems
 * s indexes slices
 * ii index pixels
*/
{
    int n2_half = n2 / 2 + 1;
    int n_pix = n1 * n2;
    int n_pix_half = n1 * n2_half;
    int batch_size = n_elems * n_slices * n_pix_half;
    int n_blocks = gridDim.x;
    int n_threads = blockDim.x;
    int n_threads_total = n_blocks * n_threads;

    int works_per_thread = batch_size / n_threads_total;
    int works_thread_remainder = batch_size % n_threads_total;

    int gidx = threadIdx.x + blockDim.x * blockIdx.x;

    // multiply location phases by scattering factors for each elements
    int n_works = works_per_thread;
    if (gidx < works_thread_remainder) {
        n_works += 1;
    }

    int work_id;  // runs over 0, 1, ..., batch_size
    for (int i = 0; i < n_works; ++i) {
        // threads is more than works
        if (works_per_thread == 0) {
            work_id = gidx;
        } else {
            if (i < works_per_thread) {
                work_id = gidx * works_per_thread + i;
            } else {
                work_id = gidx + batch_size - works_thread_remainder;
            }
        }
        
        int ss = work_id / n_pix_half;
        int ii = work_id % n_pix_half;
        int k = ss / n_slices;
        int s = ss % n_slices;

        location_phase[k*n_slices*n_pix_half + s*n_pix_half + ii].x *= scattering_factors[k*n_pix_half + ii];
        location_phase[k*n_slices*n_pix_half + s*n_pix_half + ii].y *= scattering_factors[k*n_pix_half + ii];
    }

    __syncthreads();

    // reduce sum over elements
    works_per_thread = (n_slices * n_pix_half) / n_threads_total;
    works_thread_remainder = (n_slices * n_pix_half) % n_threads_total;

    n_works = works_per_thread;
    if (gidx < works_thread_remainder) {
        n_works += 1;
    }

    // work_id runs over 0, 1, ..., n_slices * n_pix_half
    for (int i = 0; i < n_works; ++i) {
        if (works_per_thread == 0) {
            work_id = gidx;
        } else {
            if (i < works_per_thread) {
                work_id = gidx * works_per_thread + i;
            } else {
                work_id = gidx + batch_size - works_thread_remainder;
            }
        }
        int s = work_id / n_pix_half;
        int ii = work_id % n_pix_half;

        // accumulate to k = 0
        for (int k = 1; k < n_elems; ++k) {
            location_phase[s*n_pix_half + ii].x += location_phase[k*n_slices*n_pix_half + s*n_pix_half + ii].x;
            location_phase[s*n_pix_half + ii].y += location_phase[k*n_slices*n_pix_half + s*n_pix_half + ii].y;
        }
        // location_phase[s*n_pix_half + ii].x = real / (float)n_pix;
        // location_phase[s*n_pix_half + ii].y = imag / (float)n_pix;
    }
}


extern "C" int build_slices_cufft_kernel(float scattering_factors[], int n_elems,
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
    cudaMalloc((void **)&batch_data_d, sizeof(cufftReal) * batch * n_pix);
    cudaMalloc((void **)&scattering_factors_d, sizeof(cufftReal) * n_elems * n_pix_half);
    cudaMalloc((void **)&location_phase_d, sizeof(cufftComplex) * batch * n_pix_half);

    cudaMemcpy(batch_data_d, atom_histograms, sizeof(float) * batch * n_pix, cudaMemcpyHostToDevice);
    cudaMemcpy(scattering_factors_d, scattering_factors, sizeof(float) * n_elems * n_pix_half, cudaMemcpyHostToDevice);

    cufftHandle p, ip;
    /*
     * cufftResult cufftPlanMany(cufftHandle *plan, int rank, int *n, 
     *                           int *inembed, int istride, int idist, 
     *                           int *onembed, int ostride, int odist, 
     *                           cufftType type, int batch);
     */
    cufftPlanMany(&p, 2, n,
                  NULL, 1, n_pix, 
                  NULL, 1, n_pix_half,
                  CUFFT_R2C, batch);

    cufftPlanMany(&ip, 2, n,
                  NULL, 1, n_pix_half,
                  NULL, 1, n_pix,
                  CUFFT_C2R, n_slices);

    cufftExecR2C(p, (cufftReal *)batch_data_d, location_phase_d);

    int threadsPerBlock = 1024;
    int blocksPerGrid = 1024;
    convolve_fourier<<<blocksPerGrid,  threadsPerBlock>>>(location_phase_d, 
                                                          scattering_factors_d, 
                                                          n_elems, n_slices, n1, n2);

    cufftExecC2R(ip, location_phase_d, batch_data_d);
    cudaMemcpy(output, batch_data_d, sizeof(float)*n_slices*n_pix, cudaMemcpyDeviceToHost);

    cufftDestroy(p);
    cufftDestroy(ip);
    cudaFree(batch_data_d);
    cudaFree(location_phase_d);
    cudaFree(scattering_factors_d);
    return 1;
}