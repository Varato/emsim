//
// Created by Chen on 12/8/2020.
//

#ifndef EMSIM_SLICEBUILDER_H
#define EMSIM_SLICEBUILDER_H

namespace emsim {
    class SliceBuilderBatch {
    public:
        SliceBuilderBatch(float *scatteringFactors, int nElems,
                          int nSlices, int n1, int n2, float dz, float pixelSize);

        ~SliceBuilderBatch();

        void sliceGenBatch(float atomHist[], float output[]) const;

        void binAtoms(float const atomCoordinates[], unsigned nAtoms,
                      uint32_t const uniqueElemsCount[],
                      float output[]) const;

    private:
        cufftHandle m_p, m_ip;
        int m_nSlices;
        int m_n1, m_n2;  // dimensions of the slice
        float m_pixelSize, m_dz;    // pixel size of the slice
        int m_n2Half, m_nPix, m_nPixHalf;

        /* SliceBuilder does not own the following data */
        float *m_scatteringFactors;  // pre-computed scattering factors for all elements needed
        int m_nElems;                // the length of m_uniqueElements
    };
}


#endif //EMSIM_SLICEBUILDER_H
