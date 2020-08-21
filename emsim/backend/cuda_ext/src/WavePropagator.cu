//
// Created by Chen on 11/8/2020.
//

#include <cufft.h>
#include <cstdio>

#include "WavePropagator.h"
#include "WavePropagator_kernel.cuh"


namespace emsim { namespace cuda {

    WavePropagator::WavePropagator(int n1, int n2, float pixelSize, float waveLength, float relativityGamma)
            : m_n1(n1), m_n2(n2), m_nPix(n1*n2), m_pixelSize(pixelSize),
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

    void WavePropagator::spacePropagate(cufftComplex *waveFourier, float dz, cufftComplex *waveOut) const {
        waveSpacePropagate(waveFourier, m_n1, m_n2, dz, m_waveLength, m_pixelSize, waveOut);
    }

    void WavePropagator::lensPropagate(cufftComplex *wave, float cs_mm, float defocus, float aperture,
                                       cufftComplex *waveOut) const {
        if (cufftExecC2C(m_p, wave, waveOut, CUFFT_FORWARD) != CUFFT_SUCCESS) {
            fprintf(stderr, "CUFFT error: C2C plan forward executation failed\n");
        }
        waveLensPropagate(waveOut, m_n1, m_n2, m_pixelSize, m_waveLength, cs_mm, defocus, aperture, waveOut);

        if (cufftExecC2C(m_p, waveOut, waveOut, CUFFT_INVERSE) != CUFFT_SUCCESS) {
            fprintf(stderr, "CUFFT error: C2C plan forward executation failed\n");
        }
    }

    void WavePropagator::singleSlicePropagate(cufftComplex *wave, cufftReal const *slice,
                                              float dz, cufftComplex *waveOut) const {
        sliceTransmit(wave, slice, waveOut);
        if (cufftExecC2C(m_p, waveOut, waveOut, CUFFT_FORWARD) != CUFFT_SUCCESS) {
            fprintf(stderr, "CUFFT error: C2C plan forward executation failed\n");
        }
        spacePropagate(waveOut, dz, waveOut);

        if (cufftExecC2C(m_p, waveOut, waveOut, CUFFT_INVERSE) != CUFFT_SUCCESS) {
            fprintf(stderr, "CUFFT error: C2C plan backward executation failed\n");
        }
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
