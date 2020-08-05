#ifndef ROW_REDUCE_SUM_CUH_
#define ROW_REDUCE_SUM_CUH_

typedef float cufftReal;
struct float2;  // forward declaration
typedef float2 cufftComplex;


void rowReduceSum(cufftComplex *A_d, int n0, int n1, cufftComplex *output_d);

#endif