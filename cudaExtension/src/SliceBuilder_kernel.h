//
// Created by Chen on 9/8/2020.
//

#ifndef EMSIM_SLICEBUILDER_KERNEL_H
#define EMSIM_SLICEBUILDER_KERNEL_H

void binAtomsWithinSlice_(float const atomCoordinates[], unsigned nAtoms,
                         unsigned const uniqueElemsCount[], unsigned nElems,
                         unsigned n1, unsigned n2, float pixSize,
                         float output[]);

#endif //EMSIM_SLICEBUILDER_KERNEL_H
