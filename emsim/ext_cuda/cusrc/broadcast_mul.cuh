#ifndef BROAD_CAST_CUH_
#define BROAD_CAST_CUH_

typedef float cufftReal;
struct float2;  // forward declaration
typedef float2 cufftComplex;


void broadCastMul(cufftComplex *A_d, cufftReal *v_d, cufftReal a, int n0, int n1, int n2);

#endif
