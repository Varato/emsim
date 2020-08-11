////
//// Created by Chen on 11/8/2020.
////
//
//#include "WavePropagator_kernel.cuh"
//
#include <cuda_runtime.h>
#include <cufft.h>
#include <cstdio>
#include "common.cuh"


#define PI 3.14159265358979324f

__global__
void waveSliceTransmitKernel(cufftComplex *wave,  cufftReal const *slice, int nPix,
                             float waveLength, float relativityGamma, cufftComplex *waveOut)
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
            waveOut[ii].x = w_real;
            waveOut[ii].y = w_imag;
        }
        gridStartIdx += batch;
    }
}


__global__
void waveSpacePropagateKernel(cufftComplex *waveFourier,
                              int n1, int n2, float dz,
                              float waveLength, float pixSize, cufftComplex *waveOut)
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
                waveOut[ii].x = w_real / (float)nPix;
                waveOut[ii].y = w_imag / (float)nPix;
            } else {
                waveOut[ii].x = 0;
                waveOut[ii].y = 0;
            }
        }
        gridStartIdx += batch;
    }
}


__global__
void waveLensPropagateKernel(cufftComplex *waveFourier, int n1, int n2, float pixSize,
                             float waveLength, float cs_mm, float defocus, float aperture, cufftComplex *waveOut)
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
                waveOut[ii].x = w_real / (float)nPix;
                waveOut[ii].y = w_imag / (float)nPix;
            } else {
                waveOut[ii].x = 0;
                waveOut[ii].y = 0;
            }
        }
        gridStartIdx += batch;
    }
}

namespace emsim {
    void waveSliceTransmit(cufftComplex *wave, cufftReal const *slice, int nPix,
                           float waveLength, float relativityGamma, cufftComplex *waveOut) {

        unsigned blockDimX = maxThreadsPerBlock;
        if (blockDimX > nPix) blockDimX = nPix;
        auto gridDimX = (unsigned) ceilf((float) nPix / (float) blockDimX);
        gridDimX = gridDimX > 2147483647 ? 2147483647 : gridDimX;

        waveSliceTransmitKernel<<<gridDimX, blockDimX>>>(wave, slice, nPix, waveLength, relativityGamma, waveOut);
    }


    void waveSpacePropagate(cufftComplex *waveFourier,
                            int n1, int n2, float dz,
                            float waveLength, float pixSize, cufftComplex *waveOut) {
        unsigned nPix = n1 * n2;
        unsigned blockDimX = maxThreadsPerBlock;
        if (blockDimX > nPix) blockDimX = nPix;
        auto gridDimX = (unsigned) ceilf((float) nPix / (float) blockDimX);
        gridDimX = gridDimX > 2147483647 ? 2147483647 : gridDimX;

        waveSpacePropagateKernel<<<gridDimX, blockDimX>>>(waveFourier, n1, n2,
                                                          dz, waveLength, pixSize,
                                                          waveOut);
    }

    void waveLensPropagate(cufftComplex *waveFourier, int n1, int n2, float pixSize,
                             float waveLength, float cs_mm, float defocus, float aperture, cufftComplex *waveOut) {
        unsigned nPix = n1 * n2;
        unsigned blockDimX = maxThreadsPerBlock;
        if (blockDimX > nPix) blockDimX = nPix;
        auto gridDimX = (unsigned) ceilf((float) nPix / (float) blockDimX);
        gridDimX = gridDimX > 2147483647 ? 2147483647 : gridDimX;

        waveLensPropagateKernel<<<gridDimX, blockDimX>>>(waveFourier, n1, n2,
                                                         pixSize, waveLength, cs_mm, defocus, aperture,
                                                         waveOut);
    }
}
