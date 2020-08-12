#include <cstdio>
#include <cufft.h>
#include <thrust/device_vector.h>

#include "SliceBuilder.h"
#include "SliceBuilder_kernel.h"
#include "utils.h"

namespace emsim { namespace cuda {
    /*
     * SliceBuilder
     */
    SliceBuilder::SliceBuilder(float *scatteringFactors, int nElems,
                               int n1, int n2, float pixelSize)
        : m_scatteringFactors(scatteringFactors), m_nElems(nElems),
          m_pixelSize(pixelSize), m_n1(n1), m_n2(n2), m_p(0), m_ip(0)
    {
        m_n2Half = m_n2 / 2 + 1;
        m_nPix = m_n1 * m_n2;
        m_nPixHalf = m_n1 * m_n2Half;

        int n[2] = {m_n1, m_n2};
        if(cufftPlanMany(&m_p, 2, n,
                         nullptr, 1, m_nPix,
                         nullptr, 1, m_nPixHalf,
                         CUFFT_R2C, nElems) != CUFFT_SUCCESS) {
            fprintf(stderr, "CUFFT error: Plan creation failed");
        }


        if (cufftPlan2d(&m_ip, m_n1, m_n2, CUFFT_C2R) != CUFFT_SUCCESS) {
            fprintf(stderr, "CUFFT error: Plan creation failed");
        }
    }

    SliceBuilder::~SliceBuilder() {
        cufftDestroy(m_p);
        cufftDestroy(m_ip);
    }

    void SliceBuilder::sliceGen(float const slcAtomHist[], float output[]) const
    {

        thrust::device_vector<cufftComplex> locationPhase(m_nElems * m_nPixHalf);
        cufftComplex* locationPhasePtr = thrust::raw_pointer_cast(&locationPhase[0]);
        if(cufftExecR2C(m_p, (cufftReal *)slcAtomHist, locationPhasePtr) != CUFFT_SUCCESS) {
            fprintf(stderr, "SliceBuilder::sliceGen: CUFFT error: R2C execution failed\n");
        }

        broadCastMul(locationPhasePtr, m_scatteringFactors,
                     1.0f/(float)m_nPix, m_nElems, 1, m_nPixHalf);
        if (cudaGetLastError() != cudaSuccess) {
            fprintf(stderr, "CUDA error: %s", cudaGetErrorString(cudaGetLastError()));
        }
        rowReduceSum(locationPhasePtr, m_nElems, m_nPixHalf, locationPhasePtr);
        if (cudaGetLastError() != cudaSuccess) {
            fprintf(stderr, "CUDA error: %s", cudaGetErrorString(cudaGetLastError()));
        }

        if (cufftExecC2R(m_ip, locationPhasePtr, output) != CUFFT_SUCCESS) {
            fprintf(stderr, "CUFFT error: C2R plan executation failed");
        }
    }

    void SliceBuilder::binAtomsWithinSlice(float const atomCoordinates[], unsigned nAtoms,
                                           uint32_t const uniqueElemsCount[],
                                           float output[]) const
    {

        binAtomsWithinSlice_(atomCoordinates, nAtoms, uniqueElemsCount, m_nElems, m_n1, m_n2, m_pixelSize,output);
    }


    /*
     * SliceBuilderBatch
     */
    SliceBuilderBatch::SliceBuilderBatch(float *scatteringFactors, int nElems,
                                         int nSlices, int n1, int n2, float dz, float pixelSize)
        : m_scatteringFactors(scatteringFactors), m_nElems(nElems),
          m_nSlices(nSlices), m_n1(n1), m_n2(n2), m_dz(dz), m_pixelSize(pixelSize), m_p(0), m_ip(0)
    {
        m_n2Half = m_n2 / 2 + 1;
        m_nPix = m_n1 * m_n2;
        m_nPixHalf = m_n1 * m_n2Half;

        int n[2] = {m_n1, m_n2};
        if (cufftPlanMany(&m_p, 2, n,
                          nullptr, 1, m_nPix,
                          nullptr, 1, m_nPixHalf,
                          CUFFT_R2C, m_nElems * m_nSlices) != CUFFT_SUCCESS) {
            fprintf(stderr, "CUFFT error: Plan creation failed\n");
        }

        if (cufftPlanMany(&m_ip, 2, n,
                          nullptr, 1, m_nPixHalf,
                          nullptr, 1, m_nPix,
                          CUFFT_C2R, m_nSlices) != CUFFT_SUCCESS) {
            fprintf(stderr, "CUFFT error: Plan creation failed\n");
        }
    }


    SliceBuilderBatch::~SliceBuilderBatch() {
        cufftDestroy(m_p);
        cufftDestroy(m_ip);
    }

    void SliceBuilderBatch::sliceGenBatch(float *atomHist, float *output) const {

        thrust::device_vector<cufftComplex> locationPhase(m_nElems * m_nSlices * m_nPixHalf);
        cufftComplex* locationPhasePtr = thrust::raw_pointer_cast(&locationPhase[0]);

        if ( cufftExecR2C(m_p, (cufftReal *)atomHist, locationPhasePtr) != CUFFT_SUCCESS) {
            fprintf(stderr, "SliceBuilderBatch: CUFFT error: R2C plan executation failed\n");
        }

        broadCastMul(locationPhasePtr, thrust::raw_pointer_cast(m_scatteringFactors),
                     1.0f/(float)m_nPix, m_nElems, m_nSlices, m_nPixHalf);
        if (cudaGetLastError() != cudaSuccess) {
            fprintf(stderr, "SliceBuilderBatch: CUDA error: %s\n", cudaGetErrorString(cudaGetLastError()));
        }
        rowReduceSum(locationPhasePtr, m_nElems, m_nSlices*m_nPixHalf, locationPhasePtr);
        if (cudaGetLastError() != cudaSuccess) {
            fprintf(stderr, "SliceBuilderBatch: CUDA error: %s\n", cudaGetErrorString(cudaGetLastError()));
        }

        if (cufftExecC2R(m_ip, locationPhasePtr, output) != CUFFT_SUCCESS) {
            fprintf(stderr, "SliceBuilderBatch: CUFFT error: C2R plan executation failed\n");
        }
    }

    void SliceBuilderBatch::binAtoms(const float *atomCoordinates, unsigned int nAtoms,
                                     const uint32_t *uniqueElemsCount, float *output) const
    {
        binAtoms_(atomCoordinates, nAtoms,
                  uniqueElemsCount, m_nElems,
                  m_nSlices, m_n1, m_n2,
                  m_dz, m_pixelSize, m_pixelSize,
                  output);
    }
} }
