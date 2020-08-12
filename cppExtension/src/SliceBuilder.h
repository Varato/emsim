//
// Created by Chen on 12/8/2020.
//

#ifndef EMSIM_SLICEBUILDER_H
#define EMSIM_SLICEBUILDER_H

#include <fftw3.h>

namespace emsim {
    class SliceBuilderBatch {
    public:
        SliceBuilderBatch(float *scatteringFactors, int nElems,
                          int nSlices, int n1, int n2, float dz, float pixelSize);

        ~SliceBuilderBatch();

        void sliceGenBatch(float atomHist[], float output[]);

//        void binAtoms(float const atomCoordinates[], unsigned nAtoms,
//                      uint32_t const uniqueElemsCount[],
//                      float output[]) const;

        int getNSlice() const {return m_nSlices;}
        int getN1() const {return m_n1;}
        int getN2() const {return m_n2;}
    protected:
        void setScatteringFactors(float *ptr) {
            m_scatteringFactors = ptr;
        }

    private:
        int m_nSlices;
        int m_n1, m_n2;             // dimensions of the slice
        float m_pixelSize, m_dz;    // pixel size of the slice
        int m_n2Half, m_nPix, m_nPixHalf;

        /* SliceBuilder does not own the following data */
        float *m_scatteringFactors;  // pre-computed scattering factors for all elements needed
        int m_nElems;                // the length of m_uniqueElements
    };
}


#endif //EMSIM_SLICEBUILDER_H
