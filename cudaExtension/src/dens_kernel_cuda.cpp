//
// Created by Chen on 10/8/2020.
//
#include <pybind11/pybind11.h>

#include "SliceBuilder.h"


namespace py = pybind11;

uintptr_t cupyGetMemPtr(py::object const &cupyArray) {
    return cupyArray.attr("data").attr("ptr").cast<uintptr_t>();
}

int cupyGetShape(py::object const &cupyArray, unsigned dim) {
    return py::tuple(cupyArray.attr("shape"))[dim].cast<int >();
}

class SliceBuilderCuPyWrapper {
public:
    SliceBuilderCuPyWrapper(py::object scatteringFactor, float pixelSize) {
        uintptr_t scatFacPtr = cupyGetMemPtr(scatteringFactor);
        m_nElems = cupyGetShape(scatteringFactor, 0);
        m_n1 = cupyGetShape(scatteringFactor, 1);
        m_n2 = cupyGetShape(scatteringFactor, 2);
        m_sb = emsim::SliceBuilder((float *)scatFacPtr, m_nElems, m_n1, m_n2, pixelSize);
        m_scatteringFactor = std::move(scatteringFactor);
        m_cupy = py::module::import("cupy");
    }

    py::object binAtomsWithinSlice(py::object const &atomCoordinates, py::object const &uniqueElemsCount) {
        using namespace pybind11::literals;
        py::tuple shape = py::make_tuple(m_nElems, m_n1, m_n2);
        py::object output = m_cupy.attr("zeros")("shape"_a=shape, "dtype"_a=m_cupy.attr("float32"));

        uintptr_t coordPtr = cupyGetMemPtr(atomCoordinates);
        uintptr_t elemCntPtr = cupyGetMemPtr(uniqueElemsCount);
        uintptr_t outPtr = cupyGetMemPtr(output);

        unsigned nAtoms = cupyGetShape(atomCoordinates, 0);
        m_sb.binAtomsWithinSlice((float *)coordPtr, nAtoms, (unsigned *)elemCntPtr, (float *)outPtr);
        return output;
    }

    py::object sliceGen(py::object const &slcAtomHist) {
        using namespace pybind11::literals;
        py::tuple shape = py::make_tuple(m_n1, m_n2);
        py::object output = m_cupy.attr("zeros")("shape"_a=shape, "dtype"_a=m_cupy.attr("float32"));

        uintptr_t slcAtomHistPtr = cupyGetMemPtr(slcAtomHist);
        uintptr_t outPtr = cupyGetMemPtr(output);

        m_sb.sliceGen((float *)slcAtomHistPtr, (float *)outPtr);
        return output;
    };
private:
    int m_n1, m_n2, m_nElems;
    py::object m_scatteringFactor;  // to keep this py object alive by holding a reference.
    py::object m_cupy;
    emsim::SliceBuilder m_sb{};
};



PYBIND11_MODULE(dens_kernel_cuda, m) {
    py::class_<SliceBuilderCuPyWrapper>(m, "SliceBuilder")
            .def(py::init<py::object, float>())
            .def("binAtomsWithinSlice", &SliceBuilderCuPyWrapper::binAtomsWithinSlice, "bin atoms witin a single slice")
            .def("sliceGen", &SliceBuilderCuPyWrapper::sliceGen, "generate a single potential slice");

//    py::class_<emsim::SliceBuilderBatch>(m, "SliceBuilderBatch");
}

