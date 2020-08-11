//
// Created by Chen on 11/8/2020.
//

#include <memory>
#include <pybind11/pybind11.h>

#include "WavePropagator.h"

namespace py = pybind11;

struct float2;
typedef float2 cufftComplex;
typedef float cufftReal;


inline
size_t cupyGetMemPtr(py::object const &cupyArray) {
    return cupyArray.attr("data").attr("ptr").cast<size_t>();
}

inline
int cupyGetShape(py::object const &cupyArray, unsigned dim) {
    return py::tuple(cupyArray.attr("shape"))[dim].cast<int >();
}


class WavePropagatorCuPyWrapper {
public:
    WavePropagatorCuPyWrapper(unsigned n1, unsigned n2, float pixelSize, float waveLength, float relativityGamma)
        : m_n1(n1), m_n2(n2)
    {
        m_wp = std::make_unique<emsim::WavePropagator>(n1, n2, pixelSize, waveLength, relativityGamma);
        cupy = py::module::import("cupy");
    }

    py::object singleSlicePropagate(py::object const &wave, py::object const &slice, float dz) {
        using namespace pybind11::literals;
        py::tuple shape = py::make_tuple(m_n1, m_n2);
        py::object waveOut = cupy.attr("empty")("shape"_a = shape, "dtype"_a = cupy.attr("complex64"));

        size_t wavePtr = cupyGetMemPtr(wave);
        size_t slicePtr = cupyGetMemPtr(slice);
        size_t waveOutPtr = cupyGetMemPtr(waveOut);

        m_wp->singleSlicePropagate((cufftComplex *)wavePtr, (cufftReal *)slicePtr, dz, (cufftComplex *)waveOutPtr);
        return waveOut;
    }

    py::object multiSlicePropagate(py::object const &wave, py::object const &multiSlice, float dz) {
        using namespace pybind11::literals;
        py::tuple shape = py::make_tuple(m_n1, m_n2);
        py::object waveOut = cupy.attr("empty")("shape"_a = shape, "dtype"_a = cupy.attr("complex64"));

        int nSlices = cupyGetShape(multiSlice, 0);

        size_t wavePtr = cupyGetMemPtr(wave);
        size_t mSlicePtr = cupyGetMemPtr(multiSlice);
        size_t waveOutPtr = cupyGetMemPtr(waveOut);
        m_wp->multiSlicePropagate((cufftComplex *)wavePtr, (cufftReal *)mSlicePtr, nSlices, dz, (cufftComplex *)waveOutPtr);
        return waveOut;
    }

    py::object lensPropagate(py::object const &wave, float cs_mm, float defocus, float aperture) {
        using namespace pybind11::literals;
        py::tuple shape = py::make_tuple(m_n1, m_n2);
        py::object waveOut = cupy.attr("empty")("shape"_a = shape, "dtype"_a = cupy.attr("complex64"));

        size_t wavePtr = cupyGetMemPtr(wave);
        size_t waveOutPtr = cupyGetMemPtr(waveOut);

        m_wp->lensPropagate((cufftComplex *)wavePtr, cs_mm, defocus, aperture, (cufftComplex *)waveOutPtr);
        return waveOut;
    }

private:
    unsigned m_n1, m_n2;
    std::unique_ptr<emsim::WavePropagator> m_wp;
    py::object cupy;
};


PYBIND11_MODULE(em_kernel_cuda, m) {
    py::class_<WavePropagatorCuPyWrapper>(m, "WavePropagator")
            .def(py::init<unsigned, unsigned, float, float, float>())
            .def("singleslice_propagate", &WavePropagatorCuPyWrapper::singleSlicePropagate, "propagate a wave throught a single potential slice")
            .def("multislice_propagate", &WavePropagatorCuPyWrapper::multiSlicePropagate, "propagate a wave through multiple slices")
            .def("lens_propagate", &WavePropagatorCuPyWrapper::lensPropagate, "propagate a wave through lens specified by input parameters");
}
