#include <cuda_runtime.h>
#include <cufft.h>
#include <stdio.h>
#include <math.h>

__global__ void rowReduceSumKernel(cufftComplex *A, int n0, int n1, cufftComplex *output) 
/* reduce sum A (n0 by n1) over the first dimension and store the result in rows of output.
 * A must be C-contiguous.
 * blockDim.x must be power of 2.
 *
 * Notes:
 *
 * Run this kernel once cannot achieve the total reduce. Every time a block of 1st dimension blockDim.x
 * finishes computation, blockDim.x * 2 rows in A is reduced to one row.
 *
 * In general, we use a 2D grid which consists of many 2D blocks sliding from top to down and left to right until the whole
 * array is covered. Because every time the 2D grid computes, 2 * gridDim.x * blockDim.x is covered, the number of row direction
 * sliding of the grid is ceil(n0 / (2 * gridDim.x * blockDim.x)). 
 * As a result, the array A is reduced to
 *     resultRows = ceil(n0 / (2 * gridDim.x * blockDim.x)) * gridDim.x
 * rows, which is effectively how many blocks compute. The resulted rows are stored in the front of output in C-contiguous mannar.
 * So to totally reduce the array A along its first dimension, do the following:
 *
 *      reduceSum_(A, n0, n1, output);
 *      int resultRows = ceil(n0 / (2 * gridDim.x * blockDim.x)) * gridDim.x;
 *      while (resultRows > 1) {
 *          reduceSum_(output, resultRows, n1, output);
 *          resultRows = ceil(resultRows / (2 * gridDim.x * blockDim.x)) * gridDim.x;
 *      }
 */
{
    extern __shared__ cufftComplex sA[];  // size is blockDim.x * blockDim.y

    // 2D grid and 2D blocks
    int rowBatch = gridDim.x * blockDim.x * 2;
    int colBatch = gridDim.y * blockDim.y;
    int col;        // to index A(n0, n1)'s coloumn
    int row0, row1; // to index the two rows of A(n0, n1) that's being summed at the first reduction level.

    // slide rightwards
    int blockStartCol;
    int gridStartCol = 0;
    while (gridStartCol < n1) {
        blockStartCol = gridStartCol + blockDim.y * blockIdx.y;
        col = blockStartCol + threadIdx.y;

        // sliding downwards (x direction)        
        int girdXSlidingCount = 0;
        int gridStartRow = 0;
        while (gridStartRow < n0) {
            row0 =  gridStartRow + blockIdx.x * blockDim.x * 2 + threadIdx.x;
            row1 = row0 + blockDim.x;
            // copy to the shared memory and do the first level reduction.
            if (col < n1) {
                if (row0 >= n0) {
                    sA[threadIdx.x * blockDim.y + threadIdx.y].x = 0;
                    sA[threadIdx.x * blockDim.y + threadIdx.y].y = 0;


                } else if(row0 < n0 && row1 >= n0) {
                    sA[threadIdx.x * blockDim.y + threadIdx.y].x = A[row0 * n1 + col].x;
                    sA[threadIdx.x * blockDim.y + threadIdx.y].y = A[row0 * n1 + col].y;


                } else {
                    sA[threadIdx.x * blockDim.y + threadIdx.y].x = A[row0 * n1 + col].x + A[row1 * n1 + col].x;
                    sA[threadIdx.x * blockDim.y + threadIdx.y].y = A[row0 * n1 + col].y + A[row1 * n1 + col].y;
                } 
            }
            __syncthreads();

            for (int step = blockDim.x / 2; step > 0; step >>= 1) {
                if (threadIdx.x < step && col < n1) {
                    sA[threadIdx.x * blockDim.y + threadIdx.y].x += sA[(threadIdx.x + step) * blockDim.y + threadIdx.y].x;
                    sA[threadIdx.x * blockDim.y + threadIdx.y].y += sA[(threadIdx.x + step) * blockDim.y + threadIdx.y].y;
                }
                __syncthreads();
            }

            // write result to global memory
            if (threadIdx.x == 0 && col < n1) {
                output[(blockIdx.x + girdXSlidingCount * gridDim.x) * n1 + col].x = sA[threadIdx.y].x;
                output[(blockIdx.x + girdXSlidingCount * gridDim.x) * n1 + col].y = sA[threadIdx.y].y;
            }
            
            gridStartRow += rowBatch;
            girdXSlidingCount += 1;
        }

        gridStartCol += colBatch;

    }
}


void rowReduceSum(cufftComplex *A_d, int n0, int n1, cufftComplex *output_d) {
    cudaDeviceProp prop;
    if(cudaGetDeviceProperties (&prop, 0) != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(cudaGetLastError()));
        return;
    }
    int nRows = n0;
    cufftComplex *inputPtr = A_d;
    do {
        // determine block dimensions
        int rowsHalved = (int) ceilf((float)nRows/2.0f);
        int blockDimX = 1;
        while (blockDimX < rowsHalved && blockDimX <= 8) {
            blockDimX <<= 1;
        }
        int blockDimY = prop.maxThreadsPerBlock / blockDimX;
        if (n1 < blockDimY) blockDimY = n1;

        // determine grid dimensions
        int gridDimX = (int)ceilf((float)rowsHalved / (float)blockDimX);
        int gridDimY = (int)ceilf((float)n1 / (float)blockDimY);
        gridDimX = gridDimX > 65535? 65535: gridDimX;
        gridDimY = gridDimY > 65535? 65535: gridDimY; 
        dim3 grid(gridDimX, gridDimY);
        dim3 block(blockDimX, blockDimY);
    
        int threadsPerBlock = block.x * block.y;
        size_t sharedMemSize = sizeof(cufftComplex) * threadsPerBlock;

        rowReduceSumKernel<<<grid, block, sharedMemSize>>>(inputPtr, nRows, n1, output_d);
        nRows = (int)ceilf((float)nRows / (float)(2 * grid.x * block.x)) * grid.x;
    } while(nRows > 1);
}