////
//// Created by Chen on 11/8/2020.
////

#include <cufft.h>
#include <cstdio>
#include "common.cuh"


#define PI 3.14159265358979324f

__global__
void waveSliceTransmitKernel(cufftComplex *wave,  cufftReal const *slice, unsigned nPix,
                             float waveLength, float relativityGamma,
                             cufftComplex *waveOut)
/*
 * transmit the wave function in real space through one single slice.
 * waveLength is in Angstroms.
 */
{
    unsigned batch = gridDim.x * blockDim.x;
    unsigned ii;  // the global index of 2D array

    float t_real, t_imag;  // transmission function
    float w_real, w_imag;  // wave
    float factor = waveLength * relativityGamma;
    //slide rightwards
    unsigned gridStartIdx = 0;
    while(gridStartIdx < nPix) {
        ii = gridStartIdx + blockDim.x * blockIdx.x + threadIdx.x;
        if (gridStartIdx > 0)
            printf("ii = %d\n", ii);
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
void waveSpacePropagateFourierKernel(cufftComplex *waveFourier,
                                     int n1, int n2, float dz, float d1, float d2,
                                     float waveLength,
                                     cufftComplex *waveOut)
{
    unsigned batch = gridDim.x * blockDim.x;
    unsigned nPix = n1 * n2;
    float dfx = 1.0f / d1 / float(n1);
    float dfy = 1.0f / d2 / float(n2);
    //Nyquist frequency and 1/3 filter
    float fMax = 0.5f / (d1 >= d2 ? d1 : d2);
    float filter = 0.6667f * fMax;

    unsigned ii;  // the global index of 2D array
    int i, j;     // the dual index
    int is, js;   // the ifftshifted indexes, signed
    float fx, fy; // the corresponding spatial frequency to is, js
    float f2;     // the squared spatial frequency f2 = fx*fx + fy*fy

    float p_real, p_imag;  // spatial propagator
    float w_real, w_imag;  // wave
    //slide rightwards
    unsigned gridStartIdx = 0;
    while(gridStartIdx < nPix) {
        ii = gridStartIdx + blockDim.x * blockIdx.x + threadIdx.x;
        if (ii < nPix) {
            i = (int)ii / n2;
            j = (int)ii % n2;
            is = i + (i<(n1+1)/2? n1/2: -(n1+1)/2);
            js = j + (j<(n2+1)/2? n2/2: -(n2+1)/2);
            fx = ((float)is - floorf((float)n1/2.0f)) * dfx;
            fy = ((float)js - floorf((float)n2/2.0f)) * dfy;
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
void waveLensPropagateKernel(cufftComplex *waveFourier, int n1, int n2, float d1, float d2,
                             float waveLength, float cs_mm, float defocus, float aperture,
                             cufftComplex *waveOut)
{
    unsigned batch = gridDim.x * blockDim.x;
    unsigned nPix = n1 * n2;
    float dfx = 1.0f / d1 / float(n1);
    float dfy = 1.0f / d2 / float(n2);
    float fAper = aperture / waveLength;
    float c1 = 0.5f * PI * cs_mm * 1e7f * waveLength * waveLength * waveLength;
    float c2 = PI * defocus * waveLength;

    unsigned ii;  // the global index of 2D array
    int i, j;     // the dual index
    int is, js;   // the ifftshifted indexes
    float fx, fy; // the corresponding spatial frequency to is, js
    float f2;     // the squared spatial frequency f2 = fx*fx + fy*fy

    float h_real, h_imag;  // modulation transfer function
    float w_real, w_imag;  // wave
    float aberr;           // optic aberration
    unsigned gridStartIdx = 0;
    while(gridStartIdx < nPix) {
        ii = gridStartIdx + blockDim.x * blockIdx.x + threadIdx.x;
        if (ii < nPix) {
            i = (int)ii / n2;
            j = (int)ii % n2;
            is = i + (i<(n1+1)/2? n1/2: -(n1+1)/2);
            js = j + (j<(n2+1)/2? n2/2: -(n2+1)/2);
            fx = ((float)is - floorf((float)n1/2.0f)) * dfx;
            fy = ((float)js - floorf((float)n2/2.0f)) * dfy;
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

namespace emsim { namespace cuda {
    void waveSliceTransmit(cufftComplex *wave, cufftReal const *slice, int nPix,
                           float waveLength, float relativityGamma,
                           cufftComplex *waveOut) {

        unsigned blockDimX = maxThreadsPerBlock;
        if (blockDimX > nPix) blockDimX = nPix;
        auto gridDimX = (unsigned) ceilf((float) nPix / (float) blockDimX);
        gridDimX = gridDimX > 2147483647 ? 2147483647 : gridDimX;
        waveSliceTransmitKernel<<<gridDimX, blockDimX>>>(wave, slice, nPix, waveLength, relativityGamma, waveOut);
    }


    void waveSpacePropagateFourier(cufftComplex *waveFourier,
                            int n1, int n2, float dz, float d1, float d2,
                            float waveLength,
                            cufftComplex *waveOut) {
        unsigned nPix = n1 * n2;
        unsigned blockDimX = maxThreadsPerBlock;
        if (blockDimX > nPix) blockDimX = nPix;
        auto gridDimX = (unsigned) ceilf((float) nPix / (float) blockDimX);
        gridDimX = gridDimX > 2147483647 ? 2147483647 : gridDimX;

        waveSpacePropagateFourierKernel<<<gridDimX, blockDimX>>>(waveFourier, n1, n2,
                                                          dz, d1, d2, waveLength,
                                                          waveOut);
    }

    void waveLensPropagate(cufftComplex *waveFourier, int n1, int n2, float d1, float d2,
                             float waveLength, float cs_mm, float defocus, float aperture,
                             cufftComplex *waveOut) {
        unsigned nPix = n1 * n2;
        unsigned blockDimX = maxThreadsPerBlock;
        if (blockDimX > nPix) blockDimX = nPix;
        auto gridDimX = (unsigned) ceilf((float) nPix / (float) blockDimX);
        gridDimX = gridDimX > 2147483647 ? 2147483647 : gridDimX;

        waveLensPropagateKernel<<<gridDimX, blockDimX>>>(waveFourier, n1, n2,
                                                         d1, d2, waveLength, cs_mm, defocus, aperture,
                                                         waveOut);
    }
} }
