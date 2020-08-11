#include <thrust/device_vector.h>

// fabsf is cuda-standardard. no need to include.
#define LEFT_CLOSE(a, b) (((b) - (a)) < (1e-12))


__global__
void binAtomsWithinSliceKernel(float const atomCoordinates[], unsigned nAtoms,
                               uint32_t const uniqueElemsCount[], unsigned nElems,
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

    float start_coord[2] = {-pixSize * floorf((float)(n1 + 1)/2.0f), // plus for number of bin edges
                            -pixSize * floorf((float)(n2 + 1)/2.0f)};

    float end_coord[2] = {start_coord[0] + (float)n1 * pixSize,
                          start_coord[1] + (float)n2 * pixSize};

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
                    i = LEFT_CLOSE(x, end_coord[0]) ? n1 - 1 : (unsigned)floorf((x - start_coord[0]) / pixSize);
                    j = LEFT_CLOSE(y, end_coord[1]) ? n2 - 1 : (unsigned)floorf((y - start_coord[1]) / pixSize);
                    atomicAdd(output + e*n1*n2 + i*n2 + j, 1.0f);
            }
        }
        gridFinishedRows +=batch;
    }
}


__global__
void binAtomsKernel(float const atomCoordinates[], unsigned nAtoms,
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


void binAtomsWithinSlice_(float const atomCoordinates[], unsigned nAtoms,
                          uint32_t const uniqueElemsCount[], unsigned nElems,
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

    binAtomsWithinSliceKernel<<<gridDimX, blockDimX>>>(atomCoordinates, nAtoms,
                                                       uniqueElemsCount, nElems,
                                                       n1, n2, pixSize,
                                                       output);
}


void binAtoms_(float const atomCoordinates[], unsigned nAtoms,
               uint32_t const uniqueElemsCount[], unsigned nElems,
               unsigned n0, unsigned n1, unsigned n2, float d0, float d1, float d2,
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

    printf("gridDim = %d, blockDim = %d\n", gridDimX, blockDimX);
    printf("n0 = %d, n1 = %d, n2 = %d, d0 = %f, d1 = %f, d2 = %f\n", n0, n1, n2, d0, d1, d2);

    binAtomsKernel<<<gridDimX, blockDimX>>>(atomCoordinates, nAtoms,
                                            uniqueElemsCount, nElems,
                                            n0, n1, n2, d0, d1, d2,
                                            output);
}
