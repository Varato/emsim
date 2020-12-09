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
        m_wp = std::make_unique<emsim::cuda::WavePropagator>(n1, n2, pixelSize, waveLength, relativityGamma);
        cupy = py::module::import("cupy");
    }

    py::object sliceTransmit(py::object const &wave, py::object const &slice) {
        using namespace pybind11::literals;
        py::tuple shape = py::make_tuple(m_n1, m_n2);
        py::object waveOut = cupy.attr("empty")("shape"_a = shape, "dtype"_a = cupy.attr("complex64"));

        size_t wavePtr = cupyGetMemPtr(wave);
        size_t slicePtr = cupyGetMemPtr(slice);
        size_t waveOutPtr = cupyGetMemPtr(waveOut);

        m_wp->sliceTransmit((cufftComplex *)wavePtr, (cufftReal *)slicePtr, (cufftComplex *)waveOutPtr);
        return waveOut;
    }

    py::object spacePropagate(py::object const &wave, float dz) {
        using namespace pybind11::literals;
        py::tuple shape = py::make_tuple(m_n1, m_n2);
        py::object waveOut = cupy.attr("empty")("shape"_a = shape, "dtype"_a = cupy.attr("complex64"));

        size_t wavePtr = cupyGetMemPtr(wave);
        size_t waveOutPtr = cupyGetMemPtr(waveOut);
        m_wp->spacePropagate((cufftComplex *)wavePtr, dz, (cufftComplex *)waveOutPtr);
        return waveOut;
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
        py::gil_scoped_release release;
        m_wp->multiSlicePropagate((cufftComplex *)wavePtr, (cufftReal *)mSlicePtr, nSlices, dz, (cufftComplex *)waveOutPtr);
        py::gil_scoped_acquire acquire;
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
    std::unique_ptr<emsim::cuda::WavePropagator> m_wp;
    py::object cupy;
};


PYBIND11_MODULE(wave_kernel_cuda, m) {
    py::class_<WavePropagatorCuPyWrapper>(m, "WavePropagator", py::module_local())
            .def(py::init<unsigned, unsigned, float, float, float>(),
                 py::arg("n1"), py::arg("n2"),
                 py::arg("pixel_size"),
                 py::arg("wave_length"), py::arg("relativity_gamma"))
            .def("slice_transmit",
                &WavePropagatorCuPyWrapper::sliceTransmit,
                py::return_value_policy::move, 
                "transmit the wave through a single potential slice without spatial propagation.",
                py::arg("wave"), py::arg("aslice"))
            .def("space_propagate",
                &WavePropagatorCuPyWrapper::spacePropagate,
                py::return_value_policy::move,
                "propagate the wave through free space by a distance dz.",
                py::arg("wave"), py::arg("dz"))
            .def("singleslice_propagate",
                 &WavePropagatorCuPyWrapper::singleSlicePropagate,
                 py::return_value_policy::move,
                 "propagate a wave throught a single potential slice",
                 py::arg("wave"), py::arg("aslice"), py::arg("dz"))
            .def("multislice_propagate",
                 &WavePropagatorCuPyWrapper::multiSlicePropagate,
                 py::return_value_policy::move,
                 "propagate a wave through multiple slices",
                 py::arg("wave"), py::arg("slices"), py::arg("dz"))
            .def("lens_propagate",
                 &WavePropagatorCuPyWrapper::lensPropagate,
                 py::return_value_policy::move,
                 "propagate a wave through lens specified by input parameters",
                 py::arg("wave"),
                 py::arg("cs_mm"),
                 py::arg("defocus"),
                 py::arg("aperture"));
}
