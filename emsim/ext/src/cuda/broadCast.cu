#include <cuda_runtime.h>
#include <cufft.h>
#include <stdio.h>
#include <math.h>

__global__ void broadcastMulKernel(cufftComplex *A, cufftReal *v, int n0, int n1, int n2)
/*
 * A: (n0, n1, n2)
 * v: (n0, n2)
 * In-place computes A * v in a broadcasting way so that the result is in shape (n0, n1, n2). 
 * A and v must be C-contiguous
 
 * Thread sliding algorithm:
 *     view v as 1D (n0 * n2, ) vector
 *     view A as 2D (n1, n0 * n2) Matrix  (Notice transpose is needed, but it only affects how we index A)
 *     Each block sliding from up to down and covers all the rows of matrix A.
 *     Because number of threads in each block is limited, we may need multiple blocks to cover all columns.
 */
{
    extern __shared__ cufftReal s[];

    int n_cols = n0 * n2;
    // we use 1D grid and 2D blocks
    int colBatch = gridDim.x * blockDim.y;
    int rowBatch = blockDim.x;
    int col, row;  // to logically index A(n1, n0 * n2), so col also index v(n0*n2, )
    int i, k;      // to logically index A(n0, n1, n2), we don't need j because j = row
    /*
     * The conversion is:
     *     j = row
     *     i = col / n2
     *     k = col % n2 = col - n2 * i
     * The global index of A(n0, n1, n2) is
     *     i * n1 * n2 + j * n2 + k
    */

    int blockStartCol;
    int gridStartCol = 0;
    while (gridStartCol < n_cols) {
        blockStartCol = gridStartCol + blockDim.y * blockIdx.x;
        col = blockStartCol + threadIdx.y;
        
        i = col / n2; 
        k = col - n2 * i;

        // the first row in the block copy from v to shared memory
        if (threadIdx.x == 0 && col < n_cols) {
            s[threadIdx.y] = v[col];
        }
        __syncthreads();

        // sliding downwords (x direction)
        int block_works_intact = n1 / rowBatch;
        int rows_remained = n1 % rowBatch;

        int start_row = 0;
        if (col < n_cols) {
            for (int w = 0; w < block_works_intact; ++w) {            
                row = start_row + threadIdx.x;
                A[i*n1*n2 + row*n2 + k].x *= s[threadIdx.y];
                A[i*n1*n2 + row*n2 + k].y *= s[threadIdx.y];
                start_row += rowBatch;
            }

            if (threadIdx.x < rows_remained) {
                row = start_row + threadIdx.x;
                A[i*n1*n2 + row*n2 + k].x *= s[threadIdx.y];
                A[i*n1*n2 + row*n2 + k].y *= s[threadIdx.y];
            }
        }
        gridStartCol += colBatch;
    }
}


void broadCastMul(cufftComplex *A_d, cufftReal *v_d, int n0, int n1, int n2) {
    cudaDeviceProp prop;
    if(cudaGetDeviceProperties (&prop, 0) != cudaSuccess) {
        printf("cuda Cannot get device information\n");
        return;
    }
    
    int nCols = n0 * n2;
    int nRows = n1;
    int blockDimX = 1;
    while (blockDimX < nRows && blockDimX <= 32) {
        blockDimX <<= 1;
    }
    int blockDimY = prop.maxThreadsPerBlock/blockDimX;
    if (nCols < blockDimY) blockDimY = nCols;
    int gridDimX = (int)ceilf((float)nCols / (float)blockDimY);
    gridDimX = gridDimX > 2147483647 ? 2147483647 : gridDimX;
    dim3 grid(gridDimX);
    dim3 block(blockDimX, blockDimY);
    // printf("grid: (%d,). block: (%d, %d)\n", gridDimX, blockDimX, blockDimY);
    size_t sharedMemSize = sizeof(cufftReal) * block.y;

    broadcastMulKernel<<<grid, block, sharedMemSize>>>(A_d, v_d, n0, n1, n2);
}