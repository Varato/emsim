#include <thrust/device_vector.h>
#include "common.cuh"
#include "SliceBuilder_kernel.h"

// fabsf is cuda-standardard. no need to include.
#define LEFT_CLOSE(a, b) (((b) - (a)) < (1e-12))


__global__
void binAtomsOneSliceKernel(float const atomCoordinates[], unsigned nAtoms,
                               uint32_t const uniqueElemsCount[], unsigned nElems,
                               unsigned n1, unsigned n2, float d1, float d2,
                               float output[])
/*
 * Given atom coordinates within one single slice, bin the coordinates into 2D histograms
 * Logical dimensions of the arrays:
 *     atomCoordinates2d: (nAtoms, 3)
 *     uniqueElems: (nElems, )
 *     uniqueElemsCount: (nElems, )
 *     output: (nElems, n1, n2), and it must be fully filled with zeros.
 *
 *     where nAtoms = reduce_sum(uniqueElemsCount)
 *
 * Here we use 1D grid and 1D block. Each thread finds the two indices for x, y coordiates.
 */
{
    unsigned batch = gridDim.x * blockDim.x;
    unsigned row;

    float start_coord[2] = {-d1 * floorf((float)(n1 + 1)/2.0f), // plus for number of bin edges
                            -d2 * floorf((float)(n2 + 1)/2.0f)};

    float end_coord[2] = {start_coord[0] + (float)n1 * d1,
                          start_coord[1] + (float)n2 * d2};

    int e, accumulatedCount;
    float x, y;
    unsigned i, j;

    unsigned gridFinishedRows = 0;
    while(gridFinishedRows < nAtoms) {
        row = gridFinishedRows + blockIdx.x * blockDim.x + threadIdx.x;
        if (row < nAtoms) {
            accumulatedCount = 0;
            for (e = 0; e < nElems; ++e) {
                accumulatedCount += uniqueElemsCount[e];
                if (row < accumulatedCount) break;
            }
            // assume the slices are along the first dimension, so 1, 2 are x and y coordinates
            x = atomCoordinates[row * 3 + 1];
            y = atomCoordinates[row * 3 + 2];
            if (start_coord[0] <= x && x <= end_coord[0] &&
                start_coord[1] <= y && y <= end_coord[1]) {
                    i = LEFT_CLOSE(x, end_coord[0]) ? n1 - 1 : (unsigned)floorf((x - start_coord[0]) / d1);
                    j = LEFT_CLOSE(y, end_coord[1]) ? n2 - 1 : (unsigned)floorf((y - start_coord[1]) / d2);
                    atomicAdd(output + e*n1*n2 + i*n2 + j, 1.0f);
            }
        }
        gridFinishedRows +=batch;
    }
}


__global__
void binAtomsMultiSlicesKernel(float const atomCoordinates[], unsigned nAtoms,
                    uint32_t const uniqueElemsCount[], unsigned nElems,
                    unsigned n0, unsigned n1, unsigned n2, float d0, float d1, float d2,
                    float output[])
/*
 * Given atom coordinates within one single slice, bin the coordinates into 2D histograms
 * Logical dimensions of the arrays:
 *     atomCoordinates2d: (nAtoms, 3)
 *     uniqueElems: (nElems, )
 *     uniqueElemsCount: (nElems, )
 *     output: (nElems, n0, n1, n2), and it must be fully filled with zeros when passing to the function.
 *
 *     where nAtoms = reduce_sum(uniqueElemsCount)
 *
 * Here we use 1D grid and 1D block. Each thread finds the two indices for x, y, z coordiates.
 */
{
    unsigned batch = gridDim.x * blockDim.x;
    unsigned row;

    unsigned nVox = n0 * n1 * n2;

    float start_coord[3] = {-d0 * floorf((float)(n0 + 1)/2.0f), // plus 1 for number of bin edges
                            -d1 * floorf((float)(n1 + 1)/2.0f),
                            -d2 * floorf((float)(n2 + 1)/2.0f)};
    float end_coord[3] = {start_coord[0] + (float)n0 * d0,
                          start_coord[1] + (float)n1 * d1,
                          start_coord[2] + (float)n2 * d2};

    float x, y, z;
    unsigned i, j, k;
    int e, accumulatedCount;

    unsigned gridFinishedRows = 0;
    while(gridFinishedRows < nAtoms) {
        row = gridFinishedRows + blockIdx.x * blockDim.x + threadIdx.x;
        if (row < nAtoms) {

            accumulatedCount = 0;
            for (e = 0; e < nElems; ++e) {
                accumulatedCount += uniqueElemsCount[e];
                if (row < accumulatedCount) break;
            }
            // assume the slices are along the first dimension, so 1, 2 are x and y coordinates
            x = atomCoordinates[row * 3 + 0];
            y = atomCoordinates[row * 3 + 1];
            z = atomCoordinates[row * 3 + 2];
            if (start_coord[0] <= x && x <= end_coord[0] &&
                start_coord[1] <= y && y <= end_coord[1] &&
                start_coord[2] <= z && z <= end_coord[2]) {
                    i = LEFT_CLOSE(x, end_coord[0]) ? n0 - 1 : (unsigned)floorf((x - start_coord[0]) / d0);
                    j = LEFT_CLOSE(y, end_coord[1]) ? n1 - 1 : (unsigned)floorf((y - start_coord[1]) / d1);
                    k = LEFT_CLOSE(z, end_coord[2]) ? n2 - 1 : (unsigned)floorf((z - start_coord[2]) / d2);
                    atomicAdd(output + e*nVox + i*n1*n2 + j*n2 + k, 1.0f);
            }
        }
        gridFinishedRows +=batch;
    }
}


__global__
void broadcastMulKernel(cufftComplex *A, cufftReal const *v, cufftReal a, unsigned n0, unsigned n1, unsigned n2)
/*
 * A: (n0, n1, n2)
 * v: (n0, n2)
 * In-place computes a*A * v in a broadcasting way so that the result is in shape (n0, n1, n2).
 * A and v must be C-contiguous

 * Thread sliding algorithm:
 *     view v as 1D (n0 * n2, ) vector
 *     view A as 2D (n1, n0 * n2) Matrix  (Notice transpose is needed, but it only affects how we index A)
 *     Each block sliding from up to down and covers all the rows of matrix A.
 *     Because number of threads in each block is limited, we may need multiple blocks to cover all columns.
 */
{
    extern __shared__ cufftReal s[];

    unsigned nCols = n0 * n2;
    // we use 1D grid and 2D blocks
    unsigned colBatch = gridDim.x * blockDim.y;
    unsigned rowBatch = blockDim.x;
    unsigned col, row;  // to logically index A(n1, n0 * n2), so col also index v(n0*n2, )
    unsigned i, k;      // to logically index A(n0, n1, n2), we don't need j because j = row
    /*
     * The conversion is:
     *     j = row
     *     i = col / n2
     *     k = col % n2 = col - n2 * i
     * The global index of A(n0, n1, n2) is
     *     i * n1 * n2 + j * n2 + k
    */

    unsigned blockStartCol;
    unsigned gridStartCol = 0;
    while (gridStartCol < nCols) {
        blockStartCol = gridStartCol + blockDim.y * blockIdx.x;
        col = blockStartCol + threadIdx.y;

        i = col / n2;
        k = col - n2 * i;

        // the first row in the block copy from v to shared memory
        if (threadIdx.x == 0 && col < nCols) {
            s[threadIdx.y] = v[col];
        }
        __syncthreads();

        // sliding downwords (x direction)
        unsigned block_works_intact = n1 / rowBatch;
        unsigned rows_remained = n1 % rowBatch;

        unsigned start_row = 0;
        if (col < nCols) {
            for (int w = 0; w < block_works_intact; ++w) {
                row = start_row + threadIdx.x;
                A[i*n1*n2 + row*n2 + k].x *= s[threadIdx.y] * a;
                A[i*n1*n2 + row*n2 + k].y *= s[threadIdx.y] * a;
                start_row += rowBatch;
            }

            if (threadIdx.x < rows_remained) {
                row = start_row + threadIdx.x;
                A[i*n1*n2 + row*n2 + k].x *= s[threadIdx.y] * a;
                A[i*n1*n2 + row*n2 + k].y *= s[threadIdx.y] * a;
            }
        }
        gridStartCol += colBatch;
    }
}


__global__
void rowReduceSumKernel(cufftComplex *A, unsigned n0, unsigned n1, cufftComplex *output)
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
    unsigned rowBatch = gridDim.x * blockDim.x * 2;
    unsigned colBatch = gridDim.y * blockDim.y;
    unsigned col;        // to index A(n0, n1)'s coloumn
    unsigned row0, row1; // to index the two rows of A(n0, n1) that's being summed at the first reduction level.

    // slide rightwards
    unsigned blockStartCol;
    unsigned gridStartCol = 0;
    while (gridStartCol < n1) {
        blockStartCol = gridStartCol + blockDim.y * blockIdx.y;
        col = blockStartCol + threadIdx.y;

        // sliding downwards (x direction)
        unsigned girdXSlidingCount = 0;
        unsigned gridStartRow = 0;
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

            for (unsigned step = blockDim.x / 2; step > 0; step >>= 1u) {
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


namespace emsim { namespace cuda {

    void binAtomsOneSlice_(float const atomCoordinates[], unsigned nAtoms,
                              uint32_t const uniqueElemsCount[], unsigned nElems,
                              unsigned n1, unsigned n2, float d1, float d2,
                              float output[]) {
        unsigned blockDimX = maxThreadsPerBlock;
        if (blockDimX > nAtoms) blockDimX = nAtoms;
        unsigned gridDimX = (int) ceilf((float) nAtoms / (float) blockDimX);

        binAtomsOneSliceKernel<<<gridDimX, blockDimX>>>(atomCoordinates, nAtoms,
                                                           uniqueElemsCount, nElems,
                                                           n1, n2, d1, d2,
                                                           output);
    }


    void binAtomsMultiSlices_(float const atomCoordinates[], unsigned nAtoms,
                   uint32_t const uniqueElemsCount[], unsigned nElems,
                   unsigned n0, unsigned n1, unsigned n2, float d0, float d1, float d2,
                   float output[]) {
        unsigned blockDimX = maxThreadsPerBlock;
        if (blockDimX > nAtoms) blockDimX = nAtoms;
        unsigned gridDimX = (int) ceilf((float) nAtoms / (float) blockDimX);

        binAtomsMultiSlicesKernel<<<gridDimX, blockDimX>>>(atomCoordinates, nAtoms,
                                                uniqueElemsCount, nElems,
                                                n0, n1, n2, d0, d1, d2,
                                                output);
    }

    void broadCastMul_(cufftComplex *A_d, cufftReal *v_d, cufftReal a, unsigned n0, unsigned n1, unsigned n2) {

        unsigned nCols = n0 * n2;
        unsigned nRows = n1;
        unsigned blockDimX = 1;
        while (blockDimX < nRows && blockDimX <= 32) {
            blockDimX <<= 1u;
        }
        unsigned blockDimY = maxThreadsPerBlock / blockDimX;
        if (nCols < blockDimY) blockDimY = nCols;
        auto gridDimX = (unsigned) ceilf((float) nCols / (float) blockDimY);
        gridDimX = gridDimX > 2147483647 ? 2147483647 : gridDimX;
        dim3 grid(gridDimX);
        dim3 block(blockDimX, blockDimY);
        // printf("grid: (%d,). block: (%d, %d)\n", gridDimX, blockDimX, blockDimY);
        size_t sharedMemSize = sizeof(cufftReal) * block.y;

        broadcastMulKernel<<<grid, block, sharedMemSize>>>(A_d, v_d, a, n0, n1, n2);
    }


    void rowReduceSum_(cufftComplex *A_d, unsigned n0, unsigned n1, cufftComplex *output_d) {

        unsigned nRows = n0;
        cufftComplex *inputPtr = A_d;
        do {
            // determine block dimensions
            unsigned rowsHalved = (int) ceilf((float) nRows / 2.0f);
            unsigned blockDimX = 1;
            while (blockDimX < rowsHalved && blockDimX <= 8) {
                blockDimX <<= 1u;
            }
            unsigned blockDimY = maxThreadsPerBlock / blockDimX;
            if (n1 < blockDimY) blockDimY = n1;

            // determine grid dimensions
            unsigned gridDimX = (int) ceilf((float) rowsHalved / (float) blockDimX);
            unsigned gridDimY = (int) ceilf((float) n1 / (float) blockDimY);
            gridDimX = gridDimX > 65535 ? 65535 : gridDimX;
            gridDimY = gridDimY > 65535 ? 65535 : gridDimY;
            dim3 grid(gridDimX, gridDimY);
            dim3 block(blockDimX, blockDimY);

            unsigned threadsPerBlock = block.x * block.y;
            size_t sharedMemSize = sizeof(cufftComplex) * threadsPerBlock;

            rowReduceSumKernel<<<grid, block, sharedMemSize>>>(inputPtr, nRows, n1, output_d);
            nRows = (unsigned) ceilf((float) nRows / (float) (2 * grid.x * block.x)) * grid.x;
        } while (nRows > 1);
    }
} }
