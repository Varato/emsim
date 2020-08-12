//
// Created by Chen on 9/8/2020.
//

#ifndef EMSIM_SLICEBUILDER_KERNEL_H
#define EMSIM_SLICEBUILDER_KERNEL_H
namespace emsim { namespace cuda {

    void binAtomsWithinSlice_(float const atomCoordinates[], unsigned nAtoms,
                              unsigned const uniqueElemsCount[], unsigned nElems,
                              unsigned n1, unsigned n2, float pixSize,
                              float output[]);

    void binAtoms_(float const atomCoordinates[], unsigned nAtoms,
                   unsigned const uniqueElemsCount[], unsigned nElems,
                   unsigned n0, unsigned n1, unsigned n2, float d0, float d1, float d2,
                   float output[]);
} }

#endif //EMSIM_SLICEBUILDER_KERNEL_H
