//
// Created by Chen on 12/8/2020.
//
#include <fftw3.h>
#include <cstdlib>
#include <cstdio>
#include <omp.h>
#include <string>
#include "SliceBuilder.h"

namespace emsim {

    MultiSlicesBuilder::MultiSlicesBuilder(float *scatteringFactors, int nElems, int nSlices,
                                         int n1, int n2, float dz, float d1, float d2)
        : m_scatteringFactors(scatteringFactors), m_nElems(nElems), m_nSlices(nSlices),
          m_n1(n1), m_n2(n2), m_n2Half(n2/2+1), m_nPix(n1*n2), m_dz(dz), m_d1(d1), m_d2(d2)
    {
        m_nPixHalf = m_n1 * m_n2Half;
        m_dfx = 1.0f / m_d1 / (float)m_n1;
        m_dfy = 1.0f / m_d2 / (float)m_n2;
        //Nyquist frequency
        m_fmax = 0.5f / (m_d1 >= m_d2 ? m_d1 : m_d2);
    }

    MultiSlicesBuilder::~MultiSlicesBuilder() = default;

    void MultiSlicesBuilder::makeMultiSlices(float *atomHist, float *output) {
        fftwf_plan p, ip;
        fftwf_complex *data;

        data = (fftwf_complex *) fftwf_malloc(sizeof(fftwf_complex) * m_nElems * m_nSlices * m_nPixHalf);

        if(!data){
            fprintf(stderr, "fftwf_malloc failed\n");
        }

        if (!fftwf_init_threads()) {
            printf("fftw cannot work with multi threads\n");
        }

        int n[2] = {m_n1, m_n2};
        fftwf_plan_with_nthreads(omp_get_max_threads());
        p  = fftwf_plan_many_dft_r2c(2, n, m_nElems * m_nSlices,
                                     atomHist, NULL,
                                     1, m_nPix,
                                     data, NULL,
                                     1, m_nPixHalf,
                                     FFTW_ESTIMATE);

        ip = fftwf_plan_many_dft_c2r(2, n, m_nSlices,
                                     data, NULL,
                                     1, m_nPixHalf,
                                     output, NULL,
                                     1, m_nPix,
                                     FFTW_ESTIMATE);
        fftwf_execute(p);

        // Convolve and sum over elements in fourier space
        int s;
        #pragma omp parallel for
        for (s = 0; s < m_nSlices; ++s) {
            for (int ii = 0; ii < m_nPixHalf; ++ii) {
                float slices_fourier_real = 0;
                float slices_fourier_imag = 0;
                int i = ii / m_n2Half;
                int j = ii % m_n2Half;

                int is = i + (i<(m_n1+1)/2? m_n1/2: -(m_n1+1)/2);
                int js = j + (j<(m_n2+1)/2? m_n2/2: -(m_n2+1)/2);
                float fx = ((float)is - floorf((float)m_n1/2.0f)) * m_dfx;
                float fy = ((float)js - floorf((float)m_n2/2.0f)) * m_dfy;
                float f2 = fx*fx + fy*fy;

                // symmetric filtering by f_max
                if (f2 <= m_fmax * m_fmax) {
                    for (int k = 0; k < m_nElems; ++k){
                        float phase_real = data[k*m_nSlices*m_nPixHalf + s*m_nPixHalf + ii][0];
                        float phase_imag = data[k*m_nSlices*m_nPixHalf + s*m_nPixHalf + ii][1];
                        float scat_fac = m_scatteringFactors[k*m_nPixHalf + ii];
                        slices_fourier_real += phase_real * scat_fac;
                        slices_fourier_imag += phase_imag * scat_fac;
                    }

                    slices_fourier_real /= (float)m_nPix;
                    slices_fourier_imag /= (float)m_nPix;
                    data[s*m_nPixHalf + ii][0] = slices_fourier_real;
                    data[s*m_nPixHalf + ii][1] = slices_fourier_imag;
                } else {
                    data[s*m_nPixHalf + ii][0] = 0.0f;
                    data[s*m_nPixHalf + ii][1] = 0.0f;
                }
            }
        }
        fftwf_execute(ip);
        fftwf_destroy_plan(p);
        fftwf_destroy_plan(ip);
        fftwf_free(data);
    }
}
