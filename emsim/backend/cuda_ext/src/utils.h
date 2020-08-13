#ifndef EMSIM_UTILS_H
#define EMSIM_UTILS_H

typedef float cufftReal;
struct float2;  // forward declaration
typedef float2 cufftComplex;

namespace emsim { namespace cuda {
    void rowReduceSum(cufftComplex *A_d, unsigned n0, unsigned n1, cufftComplex *output_d);
    void broadCastMul(cufftComplex *A_d, cufftReal *v_d, cufftReal a, unsigned n0, unsigned n1, unsigned n2);
} }

#endif //EMSIM_UTILS_H
