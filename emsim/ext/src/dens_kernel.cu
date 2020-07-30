#include <cuda_runtime.h>
#include <cufft.h>

__global__ void convolve_fourier(cufftComplex *location_phase, 
                                 cufftReal *scattering_factors,
                                 int n_elems, int, n_slices, int n1, int n2,
                                 cufftComplex *slices_fourier) 
{

}

__host__
int build_slices_fftwf_kernel(float scattering_factors_ifftshifted[], int n_elems,
                              float atom_histograms[], int n_slices, int n1, int n2,
                              float output[])

{
    int n[2] = {n1, n2};
    int n2_half = n2 / 2 + 1;
    int n_pix = n1 * n2;
    int n_pix_half = n1 * n2_half;

    int batch = n_elems * n_slices;
    int full_mem_size = sizeof(float) * n_elems * n_slices * n1 * n2;


    cufftReal* atom_histograms_device, scattering_factors_device;
    cufftComplex* location_phase_device;
    cufftComplex* slices_fourier_device;
    cudaMalloc((void **)&atom_histograms_device, sizeof(cufftReal)*n_elems*n_slices*n_pix);
    cudaMalloc((void **)&location_phase_device, sizeof(cufftComplex)*n_elems*n_slices*n_pix_half);
    cudaMalloc((void **)&slices_fourier_device, sizeof(cufftComplex)*n_elems*n_slices*n_pix_half);
    cudaMalloc((void **)&scattering_factors_device, sizeof(cufftReal)*n_elems*n_pix);

    cudaMemcpy(atom_histograms_device, atom_histograms, sizeof(float) * batch * n_pix, cudaMemcpyHostToDevice);
    cudaMemcpy(scattering_factors_device, scattering_factors_ifftshifted, sizeof(float) * n_elems * n_pix, cudaMemcpyHostToDevice);



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
                  CUFFT_C2R, batch);

    cufftPlanMany(&ip, 2, n,
                  NULL, 1, n_pix_half,
                  NULL, 1, n_pix
                  CUFFT_R2C, n_slices);

    cufftExecR2C(p, (cufftReal *)atom_histograms_device, location_phase_device);

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    build_slices_fftwf_kernel<<<blocksPerGrid,  threadsPerBlock>>>(location_phase_device, 
                                                                   scattering_factors_device, 
                                                                   n_elems, n_slices, n1, n2,
                                                                   slices_fourier_device);

    cufftExecC2R(ip, slices_fourier, ?);
    cufftDestroy(p);
    cufftDestroy(ip);
    cudaFree(atom_histograms_device);
    cudaFree(location_phase_device);
    cudaFree(slices_fourier_device);
    cudaFree(scattering_factors_device);
}