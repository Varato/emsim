#ifndef UTILS_H_
#define UTILS_H_

typedef fftw_complex; // forward declaration

void fftshift2d(const fftw_complex *unshifted, fftw_complex *shifted, int n0, int n1);
void ifftshift2d(const fftw_complex *shifted, fftw_complex *unshifted, int n0, int n1);
void fftshift3d(const fftw_complex *unshifted, fftw_complex *shifted, int n0, int n1, int n2);
void ifftshift3d(const fftw_complex *shifted, fftw_complex *unshifted, int n0, int n1, int n2);
void fftshift2d_real(const float *in,  float *shifted, int n0, int n1);
void ifftshift2d_real(const float *in,  float *unshifted, int n0, int n1);


#endif