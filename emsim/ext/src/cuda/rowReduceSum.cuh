#ifndef ROW_REDUCE_SUM_CUH_
#define ROW_REDUCE_SUM_CUH_

#include <cufft.h>

void rowReduceSum(cufftComplex *A_d, int n0, int n1, cufftComplex *output_d);

#endif