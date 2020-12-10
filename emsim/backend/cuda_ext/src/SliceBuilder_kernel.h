//
// Created by Chen on 9/8/2020.
//

#ifndef EMSIM_SLICEBUILDER_KERNEL_H
#define EMSIM_SLICEBUILDER_KERNEL_H

typedef float cufftReal;
struct float2;  // forward declaration
typedef float2 cufftComplex;

namespace emsim { namespace cuda {

    void binAtomsWithinSlice_(float const atomCoordinates[], unsigned nAtoms,
                              unsigned const uniqueElemsCount[], unsigned nElems,
                              unsigned n1, unsigned n2, float pixSize,
                              float output[]);

    void binAtoms_(float const atomCoordinates[], unsigned nAtoms,
                   unsigned const uniqueElemsCount[], unsigned nElems,
                   unsigned n0, unsigned n1, unsigned n2, float d0, float d1, float d2,
                   float output[]);

    void broadCastMul(cufftComplex *A_d, cufftReal *v_d, cufftReal a, unsigned n0, unsigned n1, unsigned n2);
    void rowReduceSum(cufftComplex *A_d, unsigned n0, unsigned n1, cufftComplex *output_d);

} }

#endif //EMSIM_SLICEBUILDER_KERNEL_H
