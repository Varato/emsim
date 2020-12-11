#include <cstdio>
#include <cufft.h>
#include <thrust/device_vector.h>

#include "SliceBuilder.h"
#include "SliceBuilder_kernel.h"

namespace emsim { namespace cuda {
    /*
     * ---- OneSliceBuilder ----
     */
     OneSliceBuilder::OneSliceBuilder(float *scatteringFactors, int nElems,
                                      int n1, int n2, float d1, float d2)
        : m_scatteringFactors(scatteringFactors), m_nElems(nElems),
          m_n1(n1), m_n2(n2), m_d1(d1), m_d2(d2), m_n2Half(n2/2+1), m_nPix(n1*n2),
          m_p(0), m_ip(0)
    {
        m_nPixHalf = m_n1 * m_n2Half;

        int n[2] = {m_n1, m_n2};
        if(cufftPlanMany(&m_p, 2, n,
                         nullptr, 1, m_nPix,
                         nullptr, 1, m_nPixHalf,
                         CUFFT_R2C, nElems) != CUFFT_SUCCESS) {
            fprintf(stderr, "CUFFT error: Plan creation failed\n");
        }


        if (cufftPlan2d(&m_ip, m_n1, m_n2, CUFFT_C2R) != CUFFT_SUCCESS) {
            fprintf(stderr, "CUFFT error: Plan creation failed\n");
        }
    }

    OneSliceBuilder::~OneSliceBuilder() {
        cufftDestroy(m_p);
        cufftDestroy(m_ip);
    }

    void OneSliceBuilder::makeOneSlice(float const slcAtomHist[], float output[]) const
    {

        thrust::device_vector<cufftComplex> locationPhase(m_nElems * m_nPixHalf);
        cufftComplex* locationPhasePtr = thrust::raw_pointer_cast(&locationPhase[0]);
        if(cufftExecR2C(m_p, (cufftReal *)slcAtomHist, locationPhasePtr) != CUFFT_SUCCESS) {
            fprintf(stderr, "SliceBuilder::sliceGen: CUFFT error: R2C execution failed\n");
        }

        // locationPhase: (nElems, n1, n2//2+1)
        // scateringFactors: (nElems, n1, n2//2)
        broadCastMul_(locationPhasePtr, m_scatteringFactors,
                     1.0f/(float)m_nPix, m_nElems, 1, m_nPixHalf);
        
        if (cudaGetLastError() != cudaSuccess) {
            fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(cudaGetLastError()));
        }
        rowReduceSum_(locationPhasePtr, m_nElems, m_nPixHalf, locationPhasePtr);
        if (cudaGetLastError() != cudaSuccess) {
            fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(cudaGetLastError()));
        }

        if (cufftExecC2R(m_ip, locationPhasePtr, output) != CUFFT_SUCCESS) {
            fprintf(stderr, "CUFFT error: C2R plan executation failed\n");
        }
    }

    void OneSliceBuilder::binAtomsOneSlice(float const atomCoordinates[], unsigned nAtoms,
                                              uint32_t const uniqueElemsCount[],
                                              float output[]) const
    {

        binAtomsOneSlice_(atomCoordinates, nAtoms, uniqueElemsCount, m_nElems, m_n1, m_n2, m_d1, m_d2, output);
    }


    /*
     * ---- MultiSlicesBuilder ----
     */
     MultiSlicesBuilder::MultiSlicesBuilder(float *scatteringFactors, int nElems,
                                            int nSlices, int n1, int n2, float dz, float d1, float d2)
        : m_scatteringFactors(scatteringFactors), m_nElems(nElems),
          m_nSlices(nSlices), m_n1(n1), m_n2(n2), m_n2Half(n2/2+1), m_nPix(n1*n2), m_dz(dz), m_d1(d1), m_d2(d2),
          m_p(0), m_ip(0)
    {
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


    MultiSlicesBuilder::~MultiSlicesBuilder() {
        cufftDestroy(m_p);
        cufftDestroy(m_ip);
    }

    void MultiSlicesBuilder::makeMultiSlices(float *atomHist, float *output) const {

        thrust::device_vector<cufftComplex> locationPhase(m_nElems * m_nSlices * m_nPixHalf);
        cufftComplex* locationPhasePtr = thrust::raw_pointer_cast(&locationPhase[0]);

        if ( cufftExecR2C(m_p, (cufftReal *)atomHist, locationPhasePtr) != CUFFT_SUCCESS) {
            fprintf(stderr, "SliceBuilderBatch: CUFFT error: R2C plan executation failed\n");
        }

        // locationPhase: (nElems, nSlices, n1, n2//2+1)
        // scateringFactors: (nElems, n1, n2//2)
        broadCastMul_(locationPhasePtr, thrust::raw_pointer_cast(m_scatteringFactors),
                     1.0f/(float)m_nPix, m_nElems, m_nSlices, m_nPixHalf);
        
        if (cudaGetLastError() != cudaSuccess) {
            fprintf(stderr, "SliceBuilderBatch: CUDA error: %s\n", cudaGetErrorString(cudaGetLastError()));
        }
        rowReduceSum_(locationPhasePtr, m_nElems, m_nSlices*m_nPixHalf, locationPhasePtr);
        if (cudaGetLastError() != cudaSuccess) {
            fprintf(stderr, "SliceBuilderBatch: CUDA error: %s\n", cudaGetErrorString(cudaGetLastError()));
        }

        if (cufftExecC2R(m_ip, locationPhasePtr, output) != CUFFT_SUCCESS) {
            fprintf(stderr, "SliceBuilderBatch: CUFFT error: C2R plan executation failed\n");
        }
    }

    void MultiSlicesBuilder::binAtomsMultiSlices(const float *atomCoordinates, unsigned int nAtoms,
                                                 const uint32_t *uniqueElemsCount, float *output) const
    {
        binAtomsMultiSlices_(atomCoordinates, nAtoms,
                             uniqueElemsCount, m_nElems,
                             m_nSlices, m_n1, m_n2,
                             m_dz, m_d1, m_d2,
                             output);
    }
} }
