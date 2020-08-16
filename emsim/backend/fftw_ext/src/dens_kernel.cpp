//
// Created by Chen on 12/8/2020.
//

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <memory>
#include <complex>

#include "SliceBuilder.h"
#include "common.h"

namespace py = pybind11;

typedef py::array_t<float, py::array::c_style | py::array::forcecast> py_array_float_ctype;
typedef py::array_t<std::complex<float>, py::array::c_style | py::array::forcecast> py_array_complexf_cstype;
typedef float fftwf_complex[2];


class SliceBuilderBatchNumPyWrapper {
public:
    SliceBuilderBatchNumPyWrapper(py_array_float_ctype scatteringFactors,
                                  int nSlices, int n1, int n2, float dz, float pixelSize)
        : m_scatteringFactors(std::move(scatteringFactors)), m_nSlices(nSlices), m_n1(n1), m_n2(n2)
    {
        py::buffer_info info = m_scatteringFactors.request();
        auto* scatteringFactorsPtr = reinterpret_cast<float *>(info.ptr);
        int nElems = info.shape[0];
        m_sbb = std::make_unique<emsim::SliceBuilderBatch>(scatteringFactorsPtr, nElems, nSlices, n1, n2, dz, pixelSize);
    }

    py::array sliceGenbatch(py_array_float_ctype const &atomHists) {

        py::array output = make3dArray<float>(m_nSlices, m_n1, m_n2);

        float* atomHistPtr = (float *)(atomHists.request().ptr);
        float* outPtr = (float *)(output.request().ptr);

        py::gil_scoped_release release;
        m_sbb->sliceGenBatch(atomHistPtr, outPtr);
        py::gil_scoped_acquire acquire;

        return output;
    }

private:
    int m_n1, m_n2, m_nSlices;
    py_array_float_ctype m_scatteringFactors;
    std::unique_ptr<emsim::SliceBuilderBatch> m_sbb;
};


PYBIND11_MODULE(dens_kernel, m) {
    py::class_<SliceBuilderBatchNumPyWrapper>(m, "SliceBuilderBatch", py::module_local())
        .def(py::init<py_array_float_ctype, int, int, int, float, float>(),
             py::arg("scattering_factors"),
             py::arg("n_slices"), py::arg("n1"), py::arg("n2"),
             py::arg("dz"), py::arg("pixel_size"))
        .def("slice_gen_batch",
             &SliceBuilderBatchNumPyWrapper::sliceGenbatch,
             py::return_value_policy::move,
             "generate a batch of slices",
             py::arg("atom_histograms"));
}