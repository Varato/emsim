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


py::object binAtomsCupyWrapper(py::object const &atomCoordinates,
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

    emsim::cuda::binAtoms_((float *)atomCoordPtr, nAtoms,
                           (uint32_t *) elemCntPtr, nElems,
                           n0, n1, n2, d0, d1, d2,
                           (float *) outPtr);

    return output;
}


/*
 * Wrap SLiceBuilder so that it takes cupy array as input
 */
class SliceBuilderCuPyWrapper {
public:
    SliceBuilderCuPyWrapper(py::object scatteringFactor, int n1, int n2, float pixelSize)
        : m_scatteringFactors(std::move(scatteringFactor)), m_n1(n1), m_n2(n2)
    {
        uintptr_t scatFacPtr = cupyGetMemPtr(m_scatteringFactors);
        m_nElems = cupyGetShape(m_scatteringFactors, 0);
        m_sb = std::make_unique<emsim::cuda::SliceBuilder>((float *)scatFacPtr, m_nElems, m_n1, m_n2, pixelSize);
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
        m_sb->binAtomsWithinSlice((float *)coordPtr, nAtoms, (uint32_t *)elemCntPtr, (float *)outPtr);
        return output;
    }

    py::object sliceGen(py::object const &slcAtomHist) {
        using namespace pybind11::literals;
        py::tuple shape = py::make_tuple(m_n1, m_n2);
        py::object output = m_cupy.attr("empty")("shape"_a=shape, "dtype"_a=m_cupy.attr("float32"));

        uintptr_t slcAtomHistPtr = cupyGetMemPtr(slcAtomHist);
        uintptr_t outPtr = cupyGetMemPtr(output);

        m_sb->sliceGen((float *)slcAtomHistPtr, (float *)outPtr);
        return output;
    };
private:
    int m_n1, m_n2, m_nElems;
    py::object m_scatteringFactors;  // to keep this py object alive by holding a reference.
    py::object m_cupy;
    std::unique_ptr<emsim::cuda::SliceBuilder> m_sb;
};


/*
 * Wrap SliceBuilderBatch so that takes cupy array as input
 */
class SliceBuilderBatchCuPyWrapper {
public:
    SliceBuilderBatchCuPyWrapper(py::object scatteringFactors, int nSlice, int n1, int n2, float dz, float pixelSize)
        : m_scatteringFactors(std::move(scatteringFactors)), m_nSlice(nSlice), m_n1(n1), m_n2(n2)
    {
        uintptr_t scatFacPtr = cupyGetMemPtr(m_scatteringFactors);
        m_nElems = cupyGetShape(m_scatteringFactors, 0);
        m_sbb = std::make_unique<emsim::cuda::SliceBuilderBatch>((float *)scatFacPtr, m_nElems, nSlice, m_n1, m_n2, dz, pixelSize);
        m_cupy = py::module::import("cupy");
    }

    py::object binAtoms(py::object const &atomCoordinates, py::object const &uniqueElemsCount) {
        using namespace pybind11::literals;
        py::tuple shape = py::make_tuple(m_nElems, m_nSlice, m_n1, m_n2);
        py::object output = m_cupy.attr("zeros")("shape"_a=shape, "dtype"_a=m_cupy.attr("float32"));

        unsigned nAtoms = cupyGetShape(atomCoordinates, 0);
        uintptr_t atomCoordPtr = cupyGetMemPtr(atomCoordinates);
        uintptr_t elemCntPtr = cupyGetMemPtr(uniqueElemsCount);
        uintptr_t outPtr = cupyGetMemPtr(output);

        m_sbb->binAtoms((float *) atomCoordPtr, nAtoms, (uint32_t *) elemCntPtr, (float *) outPtr);
        return output;
    }

    py::object sliceGenBatch(py::object const &atomHist) {
        using namespace pybind11::literals;
        py::tuple shape = py::make_tuple(m_nSlice, m_n1, m_n2);
        py::object output = m_cupy.attr("empty")("shape"_a=shape, "dtype"_a=m_cupy.attr("float32"));

        uintptr_t atomHistPtr = cupyGetMemPtr(atomHist);
        uintptr_t outPtr = cupyGetMemPtr(output);
        m_sbb->sliceGenBatch((float *)atomHistPtr, (float *)outPtr);
        return output;
    }

private:
    py::object m_scatteringFactors;
    py::object m_cupy;
    int m_n1, m_n2, m_nElems, m_nSlice;
    std::unique_ptr<emsim::cuda::SliceBuilderBatch> m_sbb;
};



PYBIND11_MODULE(dens_kernel_cuda, m) {
    py::class_<SliceBuilderCuPyWrapper>(m, "SliceBuilder", py::module_local())
            .def(py::init<py::object, int, int, float>(),
                 py::arg("scattering_factors"), py::arg("n1"), py::arg("n1"), py::arg("pixel_size"))
            .def("bin_atoms_within_slice", &SliceBuilderCuPyWrapper::binAtomsWithinSlice, "bin atoms witin a single slice")
            .def("slice_gen", &SliceBuilderCuPyWrapper::sliceGen, "generate a single potential slice");


    py::class_<SliceBuilderBatchCuPyWrapper>(m, "SliceBuilderBatch", py::module_local())
            .def(py::init<py::object, int, int, int, float, float>(),
                 py::arg("scattering_factors"),
                 py::arg("n_slices"), py::arg("n1"), py::arg("n2"),
                 py::arg("dz"), py::arg("pixel_size"))
            .def("bin_atoms", &SliceBuilderBatchCuPyWrapper::binAtoms,
                 py::arg("atom_coordinates"), py::arg("unique_elements_counts"))
            .def("slice_gen_batch", &SliceBuilderBatchCuPyWrapper::sliceGenBatch,
                 py::arg("atom_histograms"));

    m.def("binAtoms", &binAtomsCupyWrapper,
          py::arg("atom_coordinates"),
          py::arg("unique_elements_count"),
          py::arg("n0"), py::arg("n1"), py::arg("n2"),
          py::arg("d0"), py::arg("d1"), py::arg("d2"));
}
