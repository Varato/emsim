//
// Created by Chen on 12/8/2020.
//
#include <fftw3.h>
#include <omp.h>
#include <cmath>
#include "WavePropagator.h"

#define PI 3.14159265358979324f


namespace emsim {

    WavePropagator::WavePropagator(int n1, int n2, float pixelSize, float waveLength, float relativityGamma)
        : m_n1(n1), m_n2(n2), m_nPix(n1*n2), m_pixelSize(pixelSize), m_waveLength(waveLength),
          m_relativityGamma(relativityGamma), m_p(nullptr), m_ip(nullptr)
    {
        m_dfx = 1.0f / m_pixelSize / (float)m_n1;
        m_dfy = 1.0f / m_pixelSize / (float)m_n2;
        //Nyquist frequency and 1/3 filter
        m_fmax = 0.5f / m_pixelSize;
        m_filter = 0.6667f * m_fmax;

        if (!fftwf_init_threads()) {
            printf("fftw cannot work with multi threads\n");
        }

        fftwf_plan_with_nthreads(omp_get_max_threads());

//        m_waveHolder = fftwf_alloc_complex(sizeof(fftwf_complex) * n1 * n2);
        m_p = fftwf_plan_dft_2d(n1, n2, nullptr, nullptr, FFTW_FORWARD, FFTW_ESTIMATE);
        m_ip = fftwf_plan_dft_2d(n1, n2, nullptr, nullptr, FFTW_BACKWARD, FFTW_ESTIMATE);
    }

    WavePropagator::~WavePropagator() {
        fftwf_destroy_plan(m_p);
        fftwf_destroy_plan(m_ip);
    }

    void WavePropagator::sliceTransmit(fftwf_complex *wave, const float *slice, fftwf_complex *waveOut) const {
        float factor = m_waveLength * m_relativityGamma;
        int ii;
        #pragma omp parallel for
        for (ii = 0; ii < m_nPix; ++ii){
            float pix = slice[ii];
            float t_real = cosf(pix * factor);
            float t_imag = sinf(pix * factor);
            // (wave_out[ii][0] + wave_out[ii][1] i) * (t_real + t_imag i)
            float real = wave[ii][0] * t_real - wave[ii][1] * t_imag;
            float imag = wave[ii][0] * t_imag + wave[ii][1] * t_real;
            waveOut[ii][0] = real;
            waveOut[ii][1] = imag;
        }
    }

    void WavePropagator::spacePropagate(fftwf_complex *waveFourier, float dz, fftwf_complex *waveOut) const {
        // in Fourier space, multiply the wave by spatial propagator.
        int ii;
        #pragma omp parallel for
        for (ii = 0; ii < m_nPix; ++ii){
            int i = ii / m_n2;
            int j = ii % m_n2;
            // ifftshifted indexes
            int is = i + (i<(m_n1+1)/2? m_n1/2: -(m_n1+1)/2);
            int js = j + (j<(m_n2+1)/2? m_n2/2: -(m_n2+1)/2);
            float fx = ((float)is - floorf((float)m_n1/2.0f)) * m_dfx;
            float fy = ((float)js - floorf((float)m_n2/2.0f)) * m_dfy;
            float f2 = fx*fx + fy*fy;

            if (f2 <= m_filter*m_filter) {
                // construct the spatial propagator;
                float p_real = cosf(m_waveLength * PI * dz * f2);
                float p_imag = -sinf(m_waveLength * PI * dz * f2);
                float real = waveFourier[ii][0] * p_real - waveFourier[ii][1] * p_imag;
                float imag = waveFourier[ii][0] * p_imag + waveFourier[ii][1] * p_real;
                waveOut[ii][0] = real / (float)m_nPix;
                waveOut[ii][1] = imag / (float)m_nPix;
            } else {
                waveOut[ii][0] = 0;
                waveOut[ii][1] = 0;
            }
        }
    }

    void WavePropagator::singleSlicePropagate(fftwf_complex *wave, const float *slice, float dz,
                                              fftwf_complex *waveOut) {
        sliceTransmit(wave, slice, waveOut);
        fftwf_execute_dft(m_p, waveOut, waveOut);
        spacePropagate(waveOut, dz, waveOut);
        fftwf_execute_dft(m_ip, waveOut, waveOut);
    }

    void WavePropagator::multiSlicePropagate(fftwf_complex *wave, float *multiSlices, unsigned int nSlices, float dz,
                                             fftwf_complex *waveOut) {
        // propagate through the first slice
        singleSlicePropagate(wave, multiSlices, dz, waveOut);
        for (int s = 1; s < nSlices; ++s) {
            singleSlicePropagate(waveOut, multiSlices + s * m_nPix, dz, waveOut);
        }
    }

    void WavePropagator::lensPropagate(fftwf_complex *wave, float cs_mm, float defocus, float aperture,
                                       fftwf_complex *waveOut) {
        // Just pre-calculate some factors for aberration function
        float c1 = 0.5f * PI * cs_mm * 1e7f * m_waveLength*m_waveLength*m_waveLength;
        float c2 = PI * defocus * m_waveLength;
        float f_aper = aperture / m_waveLength;

        fftwf_execute_dft(m_p, wave, waveOut);

        int ii;  // indexing pixels
        for (ii = 0; ii < m_nPix; ++ii) {
            int i = ii / m_n2;
            int j = ii % m_n2;
            // ifftshifted indexes
            int is = i + (i<(m_n1+1)/2? m_n1/2: -(m_n1+1)/2);
            int js = j + (j<(m_n2+1)/2? m_n2/2: -(m_n2+1)/2);
            float fx = ((float)is - floorf((float)m_n1/2.0f)) * m_dfx;
            float fy = ((float)js - floorf((float)m_n2/2.0f)) * m_dfy;
            float f2 = fx*fx + fy*fy;
            if (f2 <= f_aper * f_aper) {
                float aberr = c1 * f2 * f2 - c2 * f2;
                float h_real = cosf(aberr);
                float h_imag = -sinf(aberr);
                float real = waveOut[ii][0] * h_real - waveOut[ii][1] * h_imag;
                float imag = waveOut[ii][0] * h_imag + waveOut[ii][1] * h_real;
                waveOut[ii][0] = real / (float)m_nPix;
                waveOut[ii][1] = imag / (float)m_nPix;
            } else {
                waveOut[ii][0] = 0;
                waveOut[ii][1] = 0;
            }
        }
        fftwf_execute_dft(m_ip, waveOut, waveOut);
    }
}
