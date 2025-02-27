//
// Created by Chen on 11/8/2020.
//

#include <cufft.h>
#include <cstdio>

#include "WavePropagator.h"
#include "WavePropagator_kernel.cuh"


namespace emsim { namespace cuda {

    WavePropagator::WavePropagator(int n1, int n2, float d1, float d2, float waveLength, float relativityGamma)
            : m_n1(n1), m_n2(n2), m_nPix(n1*n2), m_d1(d1), m_d2(d2),
              m_waveLength(waveLength), m_relativityGamma(relativityGamma), m_p(0) {

        if (cufftPlan2d(&m_p, (int) m_n1, (int) m_n2, CUFFT_C2C) != CUFFT_SUCCESS) {
            fprintf(stderr, "CUFFT error: Plan creation failed\n");
        }
    }

    WavePropagator::~WavePropagator() {
        cufftDestroy(m_p);
    }

    void WavePropagator::sliceTransmit(cufftComplex *wave, const cufftReal *slice, cufftComplex *waveOut) const {
        waveSliceTransmit(wave, slice, m_nPix, m_waveLength, m_relativityGamma, waveOut);
    }


    void WavePropagator::spacePropagate(cufftComplex *wave, float dz, cufftComplex *waveOut) const {
        if (cufftExecC2C(m_p, wave, waveOut, CUFFT_FORWARD) != CUFFT_SUCCESS) {
            fprintf(stderr, "CUFFT error: C2C plan forward executation failed\n");
        }
        
        waveSpacePropagateFourier(waveOut, m_n1, m_n2, dz, m_d1, m_d2, m_waveLength, waveOut);

        if (cufftExecC2C(m_p, waveOut, waveOut, CUFFT_INVERSE) != CUFFT_SUCCESS) {
            fprintf(stderr, "CUFFT error: C2C plan backward executation failed\n");
        }
    }

    void WavePropagator::lensPropagate(cufftComplex *wave, float cs_mm, float defocus, float aperture,
                                       cufftComplex *waveOut) const {
        if (cufftExecC2C(m_p, wave, waveOut, CUFFT_FORWARD) != CUFFT_SUCCESS) {
            fprintf(stderr, "CUFFT error: C2C plan forward executation failed\n");
        }
        waveLensPropagate(waveOut, m_n1, m_n2, m_d1, m_d2, m_waveLength, cs_mm, defocus, aperture, waveOut);

        if (cufftExecC2C(m_p, waveOut, waveOut, CUFFT_INVERSE) != CUFFT_SUCCESS) {
            fprintf(stderr, "CUFFT error: C2C plan forward executation failed\n");
        }
    }

    void WavePropagator::singleSlicePropagate(cufftComplex *wave, cufftReal const *slice,
                                              float dz, cufftComplex *waveOut) const {
        sliceTransmit(wave, slice, waveOut);
        spacePropagate(waveOut, dz, waveOut);
    }

    void WavePropagator::multiSlicePropagate(cufftComplex *wave, cufftReal *multiSlices, unsigned int nSlices, float dz,
                                             cufftComplex *waveOut) const
    {
        // propagate through the first slice
        singleSlicePropagate(wave, multiSlices, dz, waveOut);
        for (int s = 1; s < nSlices; ++s) {
            singleSlicePropagate(waveOut, multiSlices + s * m_nPix, dz, waveOut);
        }
    }
} }
