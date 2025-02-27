//
// Created by Chen on 11/8/2020.
//

#ifndef EMSIM_WAVEPROPAGATOR_H
#define EMSIM_WAVEPROPAGATOR_H

typedef int cufftHandle;
struct float2;
typedef float2 cufftComplex;
typedef float cufftReal;

namespace emsim { namespace cuda {
    /*
     * The wave propagator in C++ side.
     * It deals with RAII for cufft.
     */
    class WavePropagator {
    public:
        WavePropagator(int n1, int n2, float d1, float d2, float waveLength, float relativityGamma);
        ~WavePropagator();

        void sliceTransmit(cufftComplex *wave,  cufftReal const *slice, cufftComplex *waveOut) const;
        void spacePropagate(cufftComplex *wave, float dz, cufftComplex *waveOut) const;
        void singleSlicePropagate(cufftComplex *wave, cufftReal const *slice, float dz, cufftComplex *waveOut) const;
        void multiSlicePropagate(cufftComplex *wave, cufftReal *multiSlices,
                                 unsigned nSlices, float dz,
                                 cufftComplex *waveOut) const;
        void lensPropagate(cufftComplex *wave,
                           float cs_mm, float defocus, float aperture,
                           cufftComplex *waveOut) const;

    private:
        cufftHandle m_p;
        int m_n1, m_n2;      // wave shape
        int m_nPix;
        float m_d1, m_d2;   // spatial sampling rate in Angstroms
        float m_waveLength;
        float m_relativityGamma;
    };
} }


#endif //EMSIM_WAVEPROPAGATOR_H
