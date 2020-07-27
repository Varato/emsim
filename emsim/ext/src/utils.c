#include <fftw3/fftw3.h>
#include <omp.h>


void fftshift2d(const fftwf_complex *in, fftw_complex *shifted, int n0, int n1)
/* fftshift2d for fftw data. n0 and n1 are logical dimensions of the array.*/
{
    int I, II;
    int i, j, ii, jj;
    #pragma omp parallel for private(i, j, ii, jj, I, II)
    for(I = 0; I < n0 * n1; ++I){
        i = I / n1;
        j = I % n1;

        ii = i + ( i<(n0+1)/2? n0/2: (-(n0+1)/2) );
        jj = j + ( j<(n1+1)/2? n1/2: (-(n1+1)/2) );

        II = ii * n1 + jj;
        
        shifted[II][0] = in[I][0];
        shifted[II][1] = in[I][1];
    }
}

void ifftshift2d(const fftwf_complex *in, fftw_complex *unshifted, int n0, int n1)
{
    int I, II;
    int i, j, ii, jj;
    #pragma omp parallel for private(i, j, ii, jj, I, II)
    for(I = 0; I < n0 * n1; ++I){
        i = I / n1;
        j = I % n1;

        ii = i + (i<(n0+1)/2? n0/2: -(n0+1)/2);
        jj = j + (j<(n1+1)/2? n1/2: -(n1+1)/2);

        II = ii * n1 + jj;

        unshifted[I][0] = in[II][0];
        unshifted[I][1] = in[II][1];
        
    }
}


void fftshift3d(const fftwf_complex *in, fftw_complex *shifted, int n0, int n1, int n2)
/* fftshift2d for fftw data. n0 and n1 are logical dimensions of the array.*/
{
    int I, II;
    int i, j, k, ii, jj, kk;
    #pragma omp parallel for private(i,j,k,ii,jj,kk,I,II)
    for(I = 0; I < n0 * n1 * n2; ++I){
        i = I / (n1*n2);
        j = (I % (n1*n2)) / n2;
        k = (I % (n1*n2)) % n2;

        ii = i + ( i<(n0+1)/2? n0/2: (-(n0+1)/2) );
        jj = j + ( j<(n1+1)/2? n1/2: (-(n1+1)/2) );
        kk = k + ( k<(n2+1)/2? n2/2: (-(n2+1)/2) );

        II = ii * n1 * n2 + jj * n2 + kk;
        
        shifted[II][0] = in[I][0];
        shifted[II][1] = in[I][1];
    }
}

void ifftshift3d(const fftwf_complex *in, fftw_complex *unshifted, int n0, int n1, int n2)
{
    int I, II;
    int i, j, k, ii, jj, kk;
    #pragma omp parallel for private(i,j,k,ii,jj,kk,I,II)
    for(I = 0; I < n0 * n1 * n2; ++I){
        i = I / (n1*n2);
        j = (I % (n1*n2)) / n2;
        k = (I % (n1*n2)) % n2;

        ii = i + ( i<(n0+1)/2? n0/2: (-(n0+1)/2) );
        jj = j + ( j<(n1+1)/2? n1/2: (-(n1+1)/2) );
        kk = k + ( k<(n2+1)/2? n2/2: (-(n2+1)/2) );

        II = ii * n1 * n2 + jj * n2 + kk;
        
        unshifted[I][0] = in[II][0];
        unshifted[I][1] = in[II][1];
    }
}


// void fftshift2d_real(const float *in,  float *shifted, int n0, int n1)
// /* fftshift2d for fftw data. n0 and n1 are logical dimensions of the array.*/
// {
//     int I, II;
//     int i, j, ii, jj;
//     #pragma omp parallel for private(i, j, ii, jj, I, II)
//     for(I = 0; I < n0 * n1; ++I){
//         i = I / n1;
//         j = I % n1;

//         ii = i + ( i<(n0+1)/2? n0/2: (-(n0+1)/2) );
//         jj = j + ( j<(n1+1)/2? n1/2: (-(n1+1)/2) );

//         II = ii * n1 + jj;
        
//         shifted[II] = in[I];
//     }
// }

// void ifftshift2d_real(const float *in,  float *unshifted, int n0, int n1)
// {
//     int I, II;
//     int i, j, ii, jj;
//     #pragma omp parallel for private(i, j, ii, jj, I, II)
//     for(I = 0; I < n0 * n1; ++I){
//         i = I / n1;
//         j = I % n1;

//         ii = i + (i<(n0+1)/2? n0/2: -(n0+1)/2);
//         jj = j + (j<(n1+1)/2? n1/2: -(n1+1)/2);

//         II = ii * n1 + jj;

//         unshifted[I] = in[II];
//     }
// }
