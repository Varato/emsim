//
// Created by Chen on 12/8/2020.
//

#ifndef EMSIM_SLICEBUILDER_H
#define EMSIM_SLICEBUILDER_H

#include <fftw3.h>
#include <tuple>

namespace emsim {

    class MultiSlicesBuilder {
    public:
        MultiSlicesBuilder(float *scatteringFactors, int nElems,
                          int nSlices, int n1, int n2, float dz, float d1, float d2);

        ~MultiSlicesBuilder();

        void makeMultiSlices(float atomHist[], float output[]);

        // count on numpy to bin atoms
        // void binAtoms(float const atomCoordinates[], unsigned nAtoms,
        //                 uint32_t const uniqueElemsCount[],
        //                 float output[]) const;

        int getNSlice() const {return m_nSlices;}
        int getN1() const {return m_n1;}
        int getN2() const {return m_n2;}
        
    private:
        int m_nSlices;
        int m_n1, m_n2;      // dimensions of the slices
        float m_dz;          // slice thickness
        float m_d1, m_d2;    // pixel size of the slices
        int m_n2Half, m_nPix, m_nPixHalf;

        /* SliceBuilder does not own the following data */
        float *m_scatteringFactors;  // pre-computed scattering factors for all elements needed
        int m_nElems;                // the length of m_uniqueElements

        /*
         * m_scatteringFactors is a c-contiguous 3D array in logical dimension (m_nElems, m_n1, m_n2 / 2 + 1).
         * The last dimension is halved because CUFFT_R2C is used here, so that we only need halved array in Fourier
         * space.
         * */
    };
}


#endif //EMSIM_SLICEBUILDER_H
