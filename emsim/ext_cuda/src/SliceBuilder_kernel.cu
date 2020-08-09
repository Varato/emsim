#include <thrust/device_vector.h>


__global__
void binAtomsWithinSliceKernel(float const atomCoordinates[],
                               unsigned const uniqueElemsCount[], unsigned nElems,
                               unsigned n1, unsigned n2, float pixSize,
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

    float x, y;
    float start_coord[2] = {-pixSize * floorf((float)(n1 + 1)/2.0f), // plus for number of bin edges
                            -pixSize * floorf((float)(n2 + 1)/2.0f)};

    unsigned gridFinishedRows;
    unsigned n, m = 0;
    unsigned i, j;
    for (int e = 0; e < nElems; ++e) {
        n = uniqueElemsCount[e];

        gridFinishedRows = 0;
        while(gridFinishedRows < n) {
            row = gridFinishedRows + blockIdx.x * blockDim.x + threadIdx.x + m;
            if (row < m + n) {
                // assume the slices are along the first dimension, so 1, 2 are x and y coordinates
                x = atomCoordinates[row * 3 + 1];
                y = atomCoordinates[row * 3 + 2];
                i = (unsigned)floorf((x - start_coord[0]) / pixSize);
                j = (unsigned)floorf((y - start_coord[1]) / pixSize);

                // ATOMIC OPERATION NEEDED
                atomicAdd(output+e*n1*n2 + i*n2 + j, 1);
                // output[e*n1*n2 + i*n2 + j] += 1;
            }
            gridFinishedRows +=batch;
        }

        m += n;
    }
}


void binAtomsWithinSlice_(float const atomCoordinates[], unsigned nAtoms,
                          unsigned const uniqueElemsCount[], unsigned nElems,
                          unsigned n1, unsigned n2, float pixSize,
                          float output[])
{
    // TODO: put this into a cudaInit function and use it globally
    cudaDeviceProp prop{};
    if(cudaGetDeviceProperties (&prop, 0) != cudaSuccess) {
        fprintf(stderr, "cuda Cannot get device information\n");
        return;
    }

    unsigned blockDimX = prop.maxThreadsPerBlock;
    if (blockDimX > nAtoms) blockDimX = nAtoms;
    unsigned gridDimX = (int)ceilf((float)nAtoms / (float)blockDimX);

    binAtomsWithinSliceKernel<<<gridDimX, blockDimX>>>(atomCoordinates,
                                                       uniqueElemsCount, nElems,
                                                       n1, n2, pixSize,
                                                       output);
}
