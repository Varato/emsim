//
// Created by Chen on 9/8/2020.
//

#ifndef EMSIM_SLICEBUILDER_H
#define EMSIM_SLICEBUILDER_H

#include <tuple>
#include <stdint.h>

typedef int cufftHandle;

namespace emsim { namespace cuda {
    /*
     * For single slice build
     */
    class OneSliceBuilder {
    public:
        OneSliceBuilder(float *scatteringFactors, int nElems,
                        int n1, int n2, float d1, float d2);
        ~OneSliceBuilder();

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
        void makeOneSlice(float const slcAtomHist[], float output[]) const;

        void binAtomsOneSlice(float const atomCoordinates[], unsigned nAtoms, 
                              uint32_t const uniqueElemsCount[],
                              float output[]) const;
        std::tuple<int, int> getDims() const {return {m_n1, m_n2};}
        std::tuple<float, float> getPixSize() const {return {m_d1, m_d2};}
        int getNElems() const {return m_nElems;}

    private:
        cufftHandle m_p, m_ip;
        int m_n1, m_n2;       // dimensions of the slice
        float m_d1, m_d2;    // pixel size of the slice
        int m_n2Half, m_nPix, m_nPixHalf;
        int m_nElems;         

        /* SliceBuilder does not own the following data */
        float* m_scatteringFactors;  // pre-computed scattering factors for all elements needed

        /*
         * m_scatteringFactors is a c-contiguous 3D array in logical dimension (m_nElems, m_n1, m_n2 / 2 + 1).
         * The last dimension is halved because CUFFT_R2C is used here, so that we only need halved array in Fourier
         * space.
         * */
    };


    /*
     * For multi-slices build
     * */
    class MultiSlicesBuilder {
    public:
        MultiSlicesBuilder(float *scatteringFactors, int nElems,
                          int nSlices, int n1, int n2, float dz, float d1, float d2);
        ~MultiSlicesBuilder();

        void makeMultiSlices(float atomHist[], float output[]) const;
        void binAtomsMultiSlices(float const atomCoordinates[], unsigned nAtoms,
                                 uint32_t const uniqueElemsCount[],
                                 float output[]) const;

    private:
        cufftHandle m_p, m_ip;
        int m_nSlices;
        int m_n1, m_n2;   // dimensions of the slice
        float m_dz;       // slice thickness
        float m_d1, m_d2; // pixel size of the slice
        int m_n2Half, m_nPix, m_nPixHalf;
        int m_nElems;                // the length of m_uniqueElements

        /* SliceBuilder does not own the following data */
        float* m_scatteringFactors;  // pre-computed scattering factors for all elements needed

        /*
         * m_scatteringFactors is a c-contiguous 3D array in logical dimension (m_nElems, m_n1, m_n2 / 2 + 1).
         * The last dimension is halved because CUFFT_R2C is used here, so that we only need halved array in Fourier
         * space.
         * */
    };
} }


#endif //EMSIM_SLICEBUILDER_H
