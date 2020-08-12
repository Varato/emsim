//
// Created by Chen on 12/8/2020.
//

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <complex>

#include "SliceBuilder.h"
#include "common.h"

namespace py = pybind11;

typedef py::array_t<float, py::array::c_style | py::array::forcecast> py_array_float_cstype;
typedef py::array_t<std::complex<float>, py::array::c_style | py::array::forcecast> py_array_complexf_cstype;
typedef float fftwf_complex[2];


class SliceBuilderBatchNumPyWrapper: public emsim::SliceBuilderBatch {
public:
    SliceBuilderBatchNumPyWrapper(py_array_float_cstype scatteringFactors,
                                  int nElems, int nSlices, int n1, int n2, float dz, float pixelSize)
        : emsim::SliceBuilderBatch(nullptr, nElems, nSlices, n1, n2, dz, pixelSize),
          m_scatteringFactors(std::move(scatteringFactors))
    {
        py::buffer_info info = m_scatteringFactors.request();
        auto* scatteringFactorsPtr = reinterpret_cast<float *>(info.ptr);
        setScatteringFactors(scatteringFactorsPtr);
    }

private:
    py_array_float_cstype m_scatteringFactors;
};


PYBIND11_MODULE(dens_kernel, m) {
    py::class_<SliceBuilderBatchNumPyWrapper>(m, "SliceBuilderBatch", py::module_local())
        .def(py::init<py_array_float_cstype, int, int, int, int, float, float>())
        .def("slice_gen_batch",
            [](SliceBuilderBatchNumPyWrapper &sb, py_array_float_cstype const &atomHists){
            int n1 = sb.getN1();
            int n2 = sb.getN2();
            int nSlices = sb.getNSlice();
            py::array output = make3dArray<float>(nSlices, n1, n2);

            float* atomHistPtr = (float *)(atomHists.request().ptr);
            float* outPtr = (float *)(output.request().ptr);

            py::print("entering sliceGenBatch\n");
            sb.sliceGenBatch(atomHistPtr, outPtr);
            return output;
        },
        py::return_value_policy::move,
        "generate a batch of slices");
}