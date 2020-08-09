#ifndef EM_KERNEL_H_
#define EM_KERNEL_H_

typedef float cufftReal;
struct float2;  // forward declaration
typedef float2 cufftComplex;


void multislice_propagate_cuda_device(cufftComplex *waveIn_d, int n1, int n2,
                                      cufftReal *slices_d, int nSlices, float pixSize, float dz,
                                      float waveLength, float relativityGamma,
                                      cufftComplex *waveOut_d);

void lens_propagate_cuda_device(cufftComplex *waveIn_d, int n1, int n2, float pixSize,
                                float waveLength, float cs_mm, float defocus, float aperture,
                                cufftComplex *waveOut_d);

#endif