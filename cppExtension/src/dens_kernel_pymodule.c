#include <Python.h>
#include <numpy/arrayobject.h>
#include <stdio.h>

#include "dens_kernel.h"
#include "def.h"


static PyObject* build_slices_fftw_wrapper(PyObject *self, PyObject *args, PyObject *kwds)
{
    PyArrayObject *scattering_factors_ifftshifted, *atom_histograms;

    static char *kwlist[] = {
        (char*)"scattering_factors_ifftshifted", // O!
        (char*)"atom_histograms",    // O!
        NULL
    };  

    int parse_result = PyArg_ParseTupleAndKeywords(args, kwds, "O!O!", kwlist, 
        &PyArray_Type, &scattering_factors_ifftshifted, 
        &PyArray_Type, &atom_histograms
    );
    if(!parse_result) Py_RETURN_NONE;

    int n_elems = (int) atom_histograms->dimensions[0];
    npy_intp dims[3] = {atom_histograms->dimensions[1], atom_histograms->dimensions[2], atom_histograms->dimensions[3]};

    PyArrayObject *slices = (PyArrayObject *)PyArray_EMPTY(3, dims, NPY_FLOAT32, 0);
    int succeeded = build_slices_fftwf_kernel((float *)scattering_factors_ifftshifted->data, n_elems,
                                              (float *)atom_histograms->data, (int)dims[0], (int)dims[1], (int)dims[2],
                                              (float *)slices->data);
    if (!succeeded) {
        Py_RETURN_NONE;
    }

    return PyArray_Return(slices);
}



/* Method table, Module definition and Module initialization function */
static PyMethodDef dens_kernel_methods[] = {
    {
        "build_slices_fourier_fftw", (PyCFunction)build_slices_fftw_wrapper, METH_VARARGS | METH_KEYWORDS,
        ""
    },
    {NULL, NULL, 0, NULL}
};


static PyModuleDef dens_kernel = {
    PyModuleDef_HEAD_INIT,
    "dens_kernel",
    "This module provides core functions buiding specimen densities",
    -1,
    dens_kernel_methods
};

PyMODINIT_FUNC PyInit_dens_kernel(void)
{
    PyObject *module;
    module = PyModule_Create(&dens_kernel);
    import_array();
    // import_ufunc();
    return module;
}