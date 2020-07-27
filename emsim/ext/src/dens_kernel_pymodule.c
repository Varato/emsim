#include <Python.h>
#include <numpy/arrayobject.h>
#include <fftw3/fftw3.h>
#include <stdio.h>

#include "dens_kernel.h"


static PyObject* build_slices_fourier_wrapper(PyObject *self, PyObject *args, PyObject *kwds)
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
    if(!parse_result) return NULL;

    int n_elems = scattering_factors_ifftshifted->dimensions[0];
    int len_x = scattering_factors_ifftshifted->dimensions[1];
    int len_y = scattering_factors_ifftshifted->dimensions[2];
    int n_slices = atom_histograms->dimensions[1];

    float *slices;
    slices = fftwf_malloc(sizeof(float) * n_slices * len_x * len_y);

    int succeeded = build_slices_fftwf_kernel((float *)scattering_factors_ifftshifted->data, n_elems,
                                              (float *)atom_histograms->data, n_slices, len_x, len_y,
                                              slices);
    if (!succeeded) {
        Py_RETURN_NONE;
    }

    npy_intp dims[3] = {n_slices, len_x, len_y};
    
    PyArrayObject *ret; // return value
    ret = (PyArrayObject *)PyArray_SimpleNewFromData(3, dims, NPY_FLOAT, slices);
    return PyArray_Return(ret);
}


/* Method table, Module definition and Module initialization function */
static PyMethodDef dens_kernel_methods[] = {
    {
        "build_slices_fourier_wrapper", (PyCFunction)build_slices_fourier_wrapper, METH_VARARGS | METH_KEYWORDS,
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