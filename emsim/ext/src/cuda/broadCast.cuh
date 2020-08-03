#ifndef BROAD_CAST_CUH_
#define BROAD_CAST_CUH_

#include <cufft.h>

void broadCastMul(cufftComplex *A_d, cufftReal *v_d, int n0, int n1, int n2);

#endif
