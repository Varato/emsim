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


class MultiSlicesBuilderNumPyWrapper {
public:
    MultiSlicesBuilderNumPyWrapper(py_array_float_ctype scatteringFactors,
                                  int nSlices, int n1, int n2, float dz, float d1, float d2)
        : m_scatteringFactors(std::move(scatteringFactors)), m_nSlices(nSlices), m_n1(n1), m_n2(n2)
    {
        py::buffer_info info = m_scatteringFactors.request();
        auto* scatteringFactorsPtr = reinterpret_cast<float *>(info.ptr);
        int nElems = info.shape[0];
        m_msb = std::make_unique<emsim::MultiSlicesBuilder>(scatteringFactorsPtr, nElems, nSlices, n1, n2, dz, d1, d2);
    }

    py::array makeMultiSlices(py_array_float_ctype const &atomHists) {

        py::array output = make3dArray<float>(m_nSlices, m_n1, m_n2);

        float* atomHistPtr = (float *)(atomHists.request().ptr);
        float* outPtr = (float *)(output.request().ptr);

        py::gil_scoped_release release;
        m_msb->makeMultiSlices(atomHistPtr, outPtr);
        py::gil_scoped_acquire acquire;

        return output;
    }

private:
    int m_n1, m_n2, m_nSlices;
    py_array_float_ctype m_scatteringFactors;
    std::unique_ptr<emsim::MultiSlicesBuilder> m_msb;
};


PYBIND11_MODULE(slice_kernel, m) {
    py::class_<MultiSlicesBuilderNumPyWrapper>(m, "MultiSlicesBuilder", py::module_local())
        .def(py::init<py_array_float_ctype, int, int, int, float, float, float>(),
             py::arg("scattering_factors"),
             py::arg("n_slices"), py::arg("n1"), py::arg("n2"),
             py::arg("dz"), py::arg("d1"), py::arg("d2"))
        .def("make_multi_slices",
             &MultiSlicesBuilderNumPyWrapper::makeMultiSlices,
             py::return_value_policy::move,
             "generate a batch of slices",
             py::arg("atom_histograms"));
}