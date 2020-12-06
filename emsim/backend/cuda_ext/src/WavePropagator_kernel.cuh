//
// Created by Chen on 11/8/2020.
//

#ifndef EMSIM_WAVEPROPAGATOR_KERNEL_CUH
#define EMSIM_WAVEPROPAGATOR_KERNEL_CUH

struct float2;
typedef float2 cufftComplex;
typedef float cufftReal;

namespace emsim { namespace cuda {

    void waveSliceTransmit(cufftComplex *wave, cufftReal const *slice, int nPix,
                           float waveLength, float relativityGamma, cufftComplex *waveOut);

    void waveSpacePropagateFourier(cufftComplex *waveFourier,
                            int n1, int n2, float dz,
                            float waveLength, float pixSize, cufftComplex *waveOut);

    void waveLensPropagate(cufftComplex *waveFourier, int n1, int n2, float pixSize,
                           float waveLength, float cs_mm, float defocus, float aperture, cufftComplex *waveOut);
} }


#endif //EMSIM_WAVEPROPAGATOR_KERNEL_CUH
