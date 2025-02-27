//
// Created by Chen on 10/8/2020.
//
#include <memory>
#include <pybind11/pybind11.h>

#include "SliceBuilder.h"
#include "SliceBuilder_kernel.h"


namespace py = pybind11;

inline
uintptr_t cupyGetMemPtr(py::object const &cupyArray) {
    return cupyArray.attr("data").attr("ptr").cast<uintptr_t>();
}

inline
int cupyGetShape(py::object const &cupyArray, unsigned dim) {
    return py::tuple(cupyArray.attr("shape"))[dim].cast<int >();
}


py::object binAtomsOneSliceCupyWrapper(py::object const &atomCoordinates,
                                       py::object const &uniqueElemsCount,
                                       unsigned n1, unsigned n2,
                                       float d1, float d2)
{
    using namespace pybind11::literals;

    py::object cupy = py::module::import("cupy");

    unsigned nElems = cupyGetShape(uniqueElemsCount, 0);
    unsigned nAtoms = cupyGetShape(atomCoordinates, 0);

    py::tuple shape = py::make_tuple(nElems, n1, n2);
    py::object output = cupy.attr("zeros")("shape"_a=shape, "dtype"_a=cupy.attr("float32"));

    uintptr_t atomCoordPtr = cupyGetMemPtr(atomCoordinates);
    uintptr_t elemCntPtr = cupyGetMemPtr(uniqueElemsCount);
    uintptr_t outPtr = cupyGetMemPtr(output);

    emsim::cuda::binAtomsOneSlice_((float *)atomCoordPtr, nAtoms,
                                   (uint32_t *) elemCntPtr, nElems,
                                   n1, n2, d1, d2,
                                   (float *) outPtr);

    return output;
}

py::object binAtomsMultiSlicesCupyWrapper(py::object const &atomCoordinates,
                               py::object const &uniqueElemsCount,
                               unsigned n0, unsigned n1, unsigned n2,
                               float d0, float d1, float d2)
{
    using namespace pybind11::literals;

    py::object cupy = py::module::import("cupy");

    unsigned nElems = cupyGetShape(uniqueElemsCount, 0);
    unsigned nAtoms = cupyGetShape(atomCoordinates, 0);

    py::tuple shape = py::make_tuple(nElems, n0, n1, n2);
    py::object output = cupy.attr("zeros")("shape"_a=shape, "dtype"_a=cupy.attr("float32"));

    uintptr_t atomCoordPtr = cupyGetMemPtr(atomCoordinates);
    uintptr_t elemCntPtr = cupyGetMemPtr(uniqueElemsCount);
    uintptr_t outPtr = cupyGetMemPtr(output);

    emsim::cuda::binAtomsMultiSlices_((float *)atomCoordPtr, nAtoms,
                           (uint32_t *) elemCntPtr, nElems,
                           n0, n1, n2, d0, d1, d2,
                           (float *) outPtr);

    return output;
}



/*
 * Wrap OneSliceBuilder so that it takes cupy array as input
 */
class OneSliceBuilderCuPyWrapper {
public:
    OneSliceBuilderCuPyWrapper(py::object scatteringFactor, int n1, int n2, float d1, float d2)
        : m_scatteringFactors(std::move(scatteringFactor)), m_n1(n1), m_n2(n2)
    {
        uintptr_t scatFacPtr = cupyGetMemPtr(m_scatteringFactors);
        m_nElems = cupyGetShape(m_scatteringFactors, 0);
        m_osb = std::make_unique<emsim::cuda::OneSliceBuilder>((float *)scatFacPtr, m_nElems, m_n1, m_n2, d1, d2);
        m_cupy = py::module::import("cupy");
    }

    py::object binAtomsOneSlice(py::object const &atomCoordinates, py::object const &uniqueElemsCount) {
        using namespace pybind11::literals;
        py::tuple shape = py::make_tuple(m_nElems, m_n1, m_n2);
        py::object output = m_cupy.attr("zeros")("shape"_a=shape, "dtype"_a=m_cupy.attr("float32"));
        uintptr_t coordPtr = cupyGetMemPtr(atomCoordinates);
        uintptr_t elemCntPtr = cupyGetMemPtr(uniqueElemsCount);
        uintptr_t outPtr = cupyGetMemPtr(output);

        unsigned nAtoms = cupyGetShape(atomCoordinates, 0);
        py::gil_scoped_release release;
        m_osb->binAtomsOneSlice((float *)coordPtr, nAtoms, (uint32_t *)elemCntPtr, (float *)outPtr);
        py::gil_scoped_acquire acquire;
        
        return output;
    }

    py::object makeOneSlice(py::object const &slcAtomHist) {
        using namespace pybind11::literals;
        py::tuple shape = py::make_tuple(m_n1, m_n2);
        py::object output = m_cupy.attr("empty")("shape"_a=shape, "dtype"_a=m_cupy.attr("float32"));

        uintptr_t slcAtomHistPtr = cupyGetMemPtr(slcAtomHist);
        uintptr_t outPtr = cupyGetMemPtr(output);

        py::gil_scoped_release release;
        m_osb->makeOneSlice((float *)slcAtomHistPtr, (float *)outPtr);
        py::gil_scoped_acquire acquire;
        return output;
    };
private:
    int m_n1, m_n2, m_nElems;
    py::object m_scatteringFactors;  // to keep this py object alive by holding a reference.
    py::object m_cupy;
    std::unique_ptr<emsim::cuda::OneSliceBuilder> m_osb;
};


/*
 * Wrap SliceBuilderBatch so that takes cupy array as input
 */
class MultiSlicesBuilderCuPyWrapper {
public:
    MultiSlicesBuilderCuPyWrapper(py::object scatteringFactors, int nSlice, int n1, int n2, float dz, float d1, float d2)
        : m_scatteringFactors(std::move(scatteringFactors)), m_nSlice(nSlice), m_n1(n1), m_n2(n2)
    {
        uintptr_t scatFacPtr = cupyGetMemPtr(m_scatteringFactors);
        m_nElems = cupyGetShape(m_scatteringFactors, 0);
        m_msb = std::make_unique<emsim::cuda::MultiSlicesBuilder>((float *)scatFacPtr, m_nElems, nSlice, m_n1, m_n2, dz, d1, d2);
        m_cupy = py::module::import("cupy");
    }

    py::object binAtomsMultiSlices(py::object const &atomCoordinates, py::object const &uniqueElemsCount) {
        using namespace pybind11::literals;
        py::tuple shape = py::make_tuple(m_nElems, m_nSlice, m_n1, m_n2);
        py::object output = m_cupy.attr("zeros")("shape"_a=shape, "dtype"_a=m_cupy.attr("float32"));

        unsigned nAtoms = cupyGetShape(atomCoordinates, 0);
        uintptr_t atomCoordPtr = cupyGetMemPtr(atomCoordinates);
        uintptr_t elemCntPtr = cupyGetMemPtr(uniqueElemsCount);
        uintptr_t outPtr = cupyGetMemPtr(output);

        m_msb->binAtomsMultiSlices((float *) atomCoordPtr, nAtoms, (uint32_t *) elemCntPtr, (float *) outPtr);
        return output;
    }

    py::object makeMultiSlices(py::object const &atomHist) {
        using namespace pybind11::literals;
        py::tuple shape = py::make_tuple(m_nSlice, m_n1, m_n2);
        py::object output = m_cupy.attr("empty")("shape"_a=shape, "dtype"_a=m_cupy.attr("float32"));

        uintptr_t atomHistPtr = cupyGetMemPtr(atomHist);
        uintptr_t outPtr = cupyGetMemPtr(output);
        m_msb->makeMultiSlices((float *)atomHistPtr, (float *)outPtr);
        return output;
    }

private:
    py::object m_scatteringFactors;
    py::object m_cupy;
    int m_n1, m_n2, m_nElems, m_nSlice;
    std::unique_ptr<emsim::cuda::MultiSlicesBuilder> m_msb;
};



PYBIND11_MODULE(slice_kernel_cuda, m) {
    py::class_<OneSliceBuilderCuPyWrapper>(m, "OneSliceBuilder", py::module_local())
            .def(py::init<py::object, int, int, float, float>(),
                 py::arg("scattering_factors"), py::arg("n1"), py::arg("n2"), py::arg("d1"), py::arg("d2"))
            .def("bin_atoms_one_slice", 
                 &OneSliceBuilderCuPyWrapper::binAtomsOneSlice, 
                 py::arg("atom_coordinates_sorted_by_elems"), py::arg("unique_elements_count"))
            .def("make_one_slice", 
                 &OneSliceBuilderCuPyWrapper::makeOneSlice, 
                 py::arg("atom_histograms_one_slice"));


    py::class_<MultiSlicesBuilderCuPyWrapper>(m, "MultiSlicesBuilder", py::module_local())
            .def(py::init<py::object, int, int, int, float, float, float>(),
                 py::arg("scattering_factors"),
                 py::arg("n_slices"), py::arg("n1"), py::arg("n2"),
                 py::arg("dz"), py::arg("d1"), py::arg("d2"))
            .def("bin_atoms_multi_slices", &MultiSlicesBuilderCuPyWrapper::binAtomsMultiSlices,
                 py::arg("atom_coordinates_sorted_by_elems"), py::arg("unique_elements_counts"))
            .def("make_multi_slices", &MultiSlicesBuilderCuPyWrapper::makeMultiSlices,
                 py::arg("atom_histograms"));


    m.def("bin_atoms_one_slice", &binAtomsOneSliceCupyWrapper,
        py::arg("atom_coordinates_sorted_by_elems"),
        py::arg("unique_elements_count"),
        py::arg("n1"), py::arg("n2"),
        py::arg("d1"), py::arg("d2"));


    m.def("bin_atoms_multi_slices", &binAtomsMultiSlicesCupyWrapper,
        py::arg("atom_coordinates_sorted_by_elems"),
        py::arg("unique_elements_count"),
        py::arg("n0"), py::arg("n1"), py::arg("n2"),
        py::arg("d0"), py::arg("d1"), py::arg("d2"));
}
