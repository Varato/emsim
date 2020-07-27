#ifndef UTILS_H_
#define UTILS_H_

typedef fftwf_complex; // forward declaration

void fftshift2d(const fftwf_complex *unshifted, fftwf_complex *shifted, int n0, int n1);
void ifftshift2d(const fftwf_complex *shifted, fftwf_complex *unshifted, int n0, int n1);
void fftshift3d(const fftwf_complex *unshifted, fftwf_complex *shifted, int n0, int n1, int n2);
void ifftshift3d(const fftwf_complex *shifted, fftwf_complex *unshifted, int n0, int n1, int n2);
void fftshift2d_real(const float *in,  float *shifted, int n0, int n1);
void ifftshift2d_real(const float *in,  float *unshifted, int n0, int n1);


#endif