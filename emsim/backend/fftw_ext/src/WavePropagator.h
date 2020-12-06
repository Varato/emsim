//
// Created by Chen on 12/8/2020.
//

#ifndef EMSIM_WAVEPROPAGATOR_H
#define EMSIM_WAVEPROPAGATOR_H

#include <fftw3.h>

namespace emsim {
    class WavePropagator {
    public:
        WavePropagator(int n1, int n2, float pixelSize, float waveLength, float relativityGamma);
        ~WavePropagator();

        void sliceTransmit(fftwf_complex *wave,  float const *slice, fftwf_complex *waveOut) const;

        void spacePropagate(fftwf_complex *wave, float dz, fftwf_complex *waveOut) const;

        void singleSlicePropagate(fftwf_complex *wave, float const *slice, float dz, fftwf_complex *waveOut);
        void multiSlicePropagate(fftwf_complex *wave, float *multiSlices,
                                 unsigned nSlices, float dz,
                                 fftwf_complex *waveOut);

        void lensPropagate(fftwf_complex *wave,
                           float cs_mm, float defocus, float aperture,
                           fftwf_complex *waveOut);
        int getN1() const {return m_n1;}
        int getN2() const {return m_n2;}

    private:
        fftwf_plan m_p, m_ip;
        int m_n1, m_n2; // wave shape
        int m_nPix;
        float m_pixelSize;   // spatial sampling rate in Angstrom
        float m_waveLength;
        float m_relativityGamma;
        float m_dfx, m_dfy, m_fmax, m_filter;

        //WavePropagator does not own the following data
    };
}


#endif //EMSIM_WAVEPROPAGATOR_H
