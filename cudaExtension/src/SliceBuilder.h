//
// Created by Chen on 9/8/2020.
//

#ifndef EMSIM_SLICEBUILDER_H
#define EMSIM_SLICEBUILDER_H

#include <tuple>

typedef int cufftHandle;

namespace emsim {
    /*
     * For single slice build
     */
    class SliceBuilder {
    public:
        SliceBuilder(float *scatteringFactors, int nElems,
                     int n1, int n2, float pixelSize);
        ~SliceBuilder();

        /*
         * This method generates a single potential slice given all coordinates of atoms within the slice.
         *
         * `slcAtomHist` is a c-contiguous 3D array in logical dimension (m_nElems, m_n1, m_n2), and the ordering of
         *      the first dimension must be exactly the same as m_scatteringFactors.
         * `output` is used to sotre the resulted potential slice, in shape (n1, n2).
         *
         * Notice that, by this implementation, the slice being building may not contain all elements that
         * m_scatteringFactors provides. In this case, the correspoinding portion of the slcAtomHist must
         * be filled with zeros.
         * */
        void sliceGen(float const slcAtomHist[], float output[]) const;

        void binAtomsWithinSlice(float const atomCoordinates[], unsigned nAtoms,
                                 uint32_t const uniqueElemsCount[],
                                 float output[]) const;
        std::tuple<int, int> getDims() const {return {m_n1, m_n2};}
        float getPixSize() const {return m_pixelSize;}
        int getNElems() const {return m_nElems;}

    private:
        cufftHandle m_p, m_ip;
        int m_n1, m_n2;  // dimensions of the slice
        float m_pixelSize;    // pixel size of the slice
        int m_n2Half, m_nPix, m_nPixHalf;

        /* SliceBuilder does not own the following data */
        float* m_scatteringFactors;  // pre-computed scattering factors for all elements needed
        int m_nElems;                // the length of m_uniqueElements

        /*
         * m_scatteringFactors is a c-contiguous 3D array in logical dimension (m_nElems, m_n1, m_n2 / 2 + 1).
         * The last dimension is halved because CUFFT_R2C is used here, so that we only need halved array in Fourier
         * space.
         * */
    };


    /*
     * For batch slice build
     * */
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
        float* m_scatteringFactors;  // pre-computed scattering factors for all elements needed
        int m_nElems;                // the length of m_uniqueElements
    };
}


#endif //EMSIM_SLICEBUILDER_H
