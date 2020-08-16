//
// Created by Chen on 12/8/2020.
//

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <complex>

#include "WavePropagator.h"
#include "common.h"

namespace py = pybind11;

typedef py::array_t<float, py::array::c_style | py::array::forcecast> py_array_float_ctype;
typedef py::array_t<std::complex<float>, py::array::c_style | py::array::forcecast> py_array_complexf_cstype;
typedef float fftwf_complex[2];


PYBIND11_MODULE(em_kernel, m) {
    py::class_<emsim::WavePropagator>(m, "WavePropagator", py::module_local())
        .def(py::init<int, int, float, float, float>(),
             py::arg("n1"), py::arg("n2"), py::arg("pixel_size"),
             py::arg("wave_length"), py::arg("relativity_gamma"))
        .def_property_readonly("n1", &emsim::WavePropagator::getN1)
        .def_property_readonly("n2", &emsim::WavePropagator::getN2)
        .def("singleslice_propagate",
            [](emsim::WavePropagator &wp, py_array_complexf_cstype const &wave,
               py_array_float_ctype const &slice, float dz) {
                int n1 = wp.getN1();
                int n2 = wp.getN2();
                py::array waveOut = make2dArray<std::complex<float>>(n1, n2);

                auto wavePtr = reinterpret_cast<fftwf_complex*>(wave.request().ptr);
                float* slicePtr = (float *)slice.request().ptr;
                auto waveOutPtr = reinterpret_cast<fftwf_complex*>(waveOut.request().ptr);

                wp.singleSlicePropagate(wavePtr, slicePtr, dz, waveOutPtr);
                return waveOut;
            },
        py::return_value_policy::move,
        "propagate the input wave through a single potential slice.",
        py::arg("wave"), py::arg("aslice"), py::arg("dz"))

        .def("multislice_propagate",
            [](emsim::WavePropagator &wp, py_array_complexf_cstype const &wave,
               py_array_float_ctype const &slices, float dz){
                int n1 = wp.getN1();
                int n2 = wp.getN2();
                py::array waveOut = make2dArray<std::complex<float>>(n1, n2);

                py::buffer_info sliceBufInfo = slices.request();
                size_t nSlices = sliceBufInfo.shape[0];
                auto wavePtr = reinterpret_cast<fftwf_complex*>(wave.request().ptr);
                auto slicesPtr = reinterpret_cast<float*>(sliceBufInfo.ptr);
                auto waveOutPtr = reinterpret_cast<fftwf_complex*>(waveOut.request().ptr);

                py::gil_scoped_release release;
                wp.multiSlicePropagate(wavePtr, slicesPtr, nSlices, dz, waveOutPtr);
                py::gil_scoped_acquire acquire;
                return waveOut;
            },
         py::return_value_policy::move,
         "propagate the input wave through multiple potential slices.",
             py::arg("wave"), py::arg("slices"), py::arg("dz"))

        .def("lens_propagate",
            [](emsim::WavePropagator &wp, py_array_complexf_cstype const &wave,
               float cs_mm, float defocus, float aperture){
                int n1 = wp.getN1();
                int n2 = wp.getN2();
                py::array waveOut = make2dArray<std::complex<float>>(n1, n2);
                auto wavePtr = reinterpret_cast<fftwf_complex*>(wave.request().ptr);
                auto waveOutPtr = reinterpret_cast<fftwf_complex*>(waveOut.request().ptr);
                py::gil_scoped_release release;
                wp.lensPropagate(wavePtr, cs_mm, defocus, aperture, waveOutPtr);
                py::gil_scoped_acquire acquire;
                return waveOut;
        },
         py::return_value_policy::move,
         "propagate the input wave through a lens specified by the input parameters.",
         py::arg("wave"), py::arg("cs_mm"), py::arg("defocus"), py::arg("aperture"));
}