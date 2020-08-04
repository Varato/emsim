#include <cufft.h>
#include <math.h>
#include <stdio.h>

#define PI 3.14159265359f

__global__ void waveSliceTransmit(cufftComplex *wave,  cufftReal *slice, int nPix, 
                                  float waveLength, float relativityGamma)
/* 
 * transmit the wave function in real space through one single slice.
 * waveLength is in Angstroms.
 */
{
    int batch = gridDim.x * blockDim.x;
    int ii;  // the global index of 2D array

    float t_real, t_imag;  // transmission function
    float w_real, w_imag;  // wave
    float factor = waveLength * relativityGamma;
    //slide rightwards
    int gridStartIdx = 0;
    while(gridStartIdx < nPix) {
        ii = gridStartIdx + blockDim.x * blockIdx.x + threadIdx.x;
        if (ii < nPix) {
            t_real = cosf(slice[ii] * factor);
            t_imag = sinf(slice[ii] * factor);
            w_real = wave[ii].x * t_real - wave[ii].y * t_imag;
            w_imag = wave[ii].x * t_imag + wave[ii].y * t_real;
            wave[ii].x = w_real;
            wave[ii].y = w_imag;
        }
        gridStartIdx += batch;
    }
}


__global__ void waveSpacePropagate(cufftComplex *waveFourier, 
                                   int n1, int n2, float dz,
                                   float waveLength, float pixSize)
{
    int batch = gridDim.x * blockDim.x;
    int nPix = n1 * n2;
    float dfx = 1.0f / pixSize / float(n1);
    float dfy = 1.0f / pixSize / float(n2);
    //Nyquist frequency and 1/3 filter
    float fMax = 0.5f / pixSize;
    float filter = 0.6667f * fMax;

    int ii;       // the global index of 2D array
    int i, j;     // the dual index
    int is, js;   // the ifftshifted indexes
    float fx, fy; // the corresponding spatial frequency to is, js
    float f2;     // the squared spatial frequency f2 = fx*fx + fy*fy

    float p_real, p_imag;  // spatial propagator
    float w_real, w_imag;  // wave
    //slide rightwards
    int gridStartIdx = 0;
    while(gridStartIdx < nPix) {
        ii = gridStartIdx + blockDim.x * blockIdx.x + threadIdx.x;
        if (ii < nPix) {
            i = ii / n2;
            j = ii % n2;
            is = i + (i<(n1+1)/2? n1/2: -(n1+1)/2);
            js = j + (j<(n2+1)/2? n2/2: -(n2+1)/2);
            fx = (float)(is - n1/2) * dfx;
            fy = (float)(js - n2/2) * dfy;
            f2 = fx*fx + fy*fy;
            if (f2 <= filter * filter) {
                p_real = cosf(waveLength * PI * dz * f2);
                p_imag = -sinf(waveLength * PI * dz * f2);
                w_real = waveFourier[ii].x * p_real - waveFourier[ii].y * p_imag;
                w_imag = waveFourier[ii].x * p_imag + waveFourier[ii].y * p_real;
                waveFourier[ii].x = w_real / (float)nPix;
                waveFourier[ii].y = w_imag / (float)nPix;
            } else {
                waveFourier[ii].x = 0;
                waveFourier[ii].y = 0;
            }
        }
        gridStartIdx += batch;
    }
}


__global__ void waveLensPropagate(cufftComplex *waveFourier, int n1, int n2, float pixSize, 
                                  float waveLength, float cs_mm, float defocus, float aperture)
{
    int batch = gridDim.x * blockDim.x;
    int nPix = n1 * n2;
    float dfx = 1.0f / pixSize / float(n1);
    float dfy = 1.0f / pixSize / float(n2);
    float fAper = aperture / waveLength;
    float c1 = 0.5f * PI * cs_mm * 1e7f * waveLength * waveLength * waveLength;
    float c2 = PI * defocus * waveLength;

    int ii;       // the global index of 2D array
    int i, j;     // the dual index
    int is, js;   // the ifftshifted indexes
    float fx, fy; // the corresponding spatial frequency to is, js
    float f2;     // the squared spatial frequency f2 = fx*fx + fy*fy

    float h_real, h_imag;  // modulation transfer function
    float w_real, w_imag;  // wave
    float aberr;           // optic aberration
    int gridStartIdx = 0;
    while(gridStartIdx < nPix) {
        ii = gridStartIdx + blockDim.x * blockIdx.x + threadIdx.x;
        if (ii < nPix) {
            i = ii / n2;
            j = ii % n2;
            is = i + (i<(n1+1)/2? n1/2: -(n1+1)/2);
            js = j + (j<(n2+1)/2? n2/2: -(n2+1)/2);
            fx = (float)(is - n1/2) * dfx;
            fy = (float)(js - n2/2) * dfy;
            f2 = fx*fx + fy*fy;
            if (f2 <= fAper * fAper) {
                aberr = c1 * f2 * f2 - c2 * f2;
                h_real = cosf(aberr);
                h_imag = -sinf(aberr);
                w_real = waveFourier[ii].x * h_real - waveFourier[ii].y * h_imag;
                w_imag = waveFourier[ii].x * h_imag + waveFourier[ii].y * h_real;
                waveFourier[ii].x = w_real / (float)nPix;
                waveFourier[ii].y = w_imag / (float)nPix;
            } else {
                waveFourier[ii].x = 0;
                waveFourier[ii].y = 0;
            }
        }
        gridStartIdx += batch;
    }
}


extern "C" void multislice_propagate_cuda_device(cufftComplex *waveIn_d, int n1, int n2,
                                                 cufftReal *slices_d, int nSlices, float pixSize, float dz,
                                                 float waveLength, float relativityGamma,
                                                 cufftComplex *waveOut_d)
{
    cufftHandle p;
    if(cufftPlan2d(&p, n1, n2, CUFFT_C2C) != CUFFT_SUCCESS) {
        fprintf(stderr, "CUFFT error: Plan creation failed");
    }

    size_t memSize = n1 * n2 * sizeof(cufftComplex);
    if(cudaMemcpy(waveOut_d, waveIn_d, memSize, cudaMemcpyDeviceToDevice) != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(cudaGetLastError()));
    }

    cudaDeviceProp prop;
    if(cudaGetDeviceProperties (&prop, 0) != cudaSuccess) {
        fprintf(stderr, "cuda Cannot get device information\n");
    }
    int nPix = n1 * n2;
    int blockDimX = prop.maxThreadsPerBlock;
    if (blockDimX > nPix) blockDimX = nPix;
    int gridDimX = (int)ceilf((float)nPix / (float)blockDimX);
    gridDimX = gridDimX > 2147483647 ? 2147483647 : gridDimX;

    for (int s = 0; s < nSlices; ++s) {
        waveSliceTransmit<<<gridDimX, blockDimX>>>(waveOut_d, slices_d + s, nPix, waveLength, relativityGamma);
        if (cufftExecC2C(p, waveOut_d, waveOut_d, CUFFT_FORWARD) != CUFFT_SUCCESS) {
            fprintf(stderr, "CUFFT error: C2C plan forward executation failed");
        }

        waveSpacePropagate<<<gridDimX, blockDimX>>>(waveOut_d, n1, n2, dz, waveLength, pixSize);

        if (cufftExecC2C(p, waveOut_d, waveOut_d, CUFFT_INVERSE) != CUFFT_SUCCESS) {
            fprintf(stderr, "CUFFT error: C2C plan backward executation failed");
        }
    }

    cufftDestroy(p);
}


extern "C" void lens_propagate_cuda_device(cufftComplex *waveIn_d, int n1, int n2, float pixSize,
                                           float waveLength, float cs_mm, float defocus, float aperture,
                                           cufftComplex *waveOut_d)
{
    cufftHandle p;
    if(cufftPlan2d(&p, n1, n2, CUFFT_C2C) != CUFFT_SUCCESS) {
        fprintf(stderr, "CUFFT error: Plan creation failed");
    }

    if (cufftExecC2C(p, waveIn_d, waveOut_d, CUFFT_FORWARD) != CUFFT_SUCCESS) {
        fprintf(stderr, "CUFFT error: C2C plan forward executation failed");
    }
    
    cudaDeviceProp prop;
    if(cudaGetDeviceProperties (&prop, 0) != cudaSuccess) {
        fprintf(stderr, "cuda Cannot get device information\n");
    }

    int nPix = n1 * n2;
    int blockDimX = prop.maxThreadsPerBlock;
    if (blockDimX > nPix) blockDimX = nPix;
    int gridDimX = (int)ceilf((float)nPix / (float)blockDimX);
    gridDimX = gridDimX > 2147483647 ? 2147483647 : gridDimX;

    waveLensPropagate<<<gridDimX, blockDimX>>>(waveOut_d, n1, n2, pixSize, waveLength, cs_mm, defocus, aperture);
    if (cudaGetLastError() != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s", cudaGetErrorString(cudaGetLastError()));
    }

    if (cufftExecC2C(p, waveOut_d, waveOut_d, CUFFT_INVERSE) != CUFFT_SUCCESS) {
        fprintf(stderr, "CUFFT error: C2C plan forward executation failed");
    }

    cufftDestroy(p);
}