#include <Python.h>
#include <numpy/arrayobject.h>
#include <fftw3/fftw3.h>
#include <stdio.h>

#include "em_kernel.h"
#include "def.h"


static PyObject* multislice_propagate_fftw_wrapper(PyObject *self, PyObject *args, PyObject *kwds)
{
    PyArrayObject *wave_in, *slices;
    float pixel_size, dz, wave_length, relativity_gamma;

    static char *kwlist[] = {
        (char*)"wave_in",          // O!
        (char*)"slices",           // O!
        (char*)"pixel_size",       // f
        (char*)"dz",
        (char*)"wave_length",
        (char*)"relativity_gamma",
        NULL
    };  

    int parse_result = PyArg_ParseTupleAndKeywords(args, kwds, "O!O!ffff", kwlist, 
        &PyArray_Type, &wave_in, 
        &PyArray_Type, &slices,
        &pixel_size, &dz, &wave_length, &relativity_gamma
    );
    if(!parse_result) Py_RETURN_NONE;

    int n_slices = (int) slices->dimensions[0];
    npy_intp dims[2] = {(int) slices->dimensions[1], (int) slices->dimensions[2]};


    PyArrayObject *wave_out = (PyArrayObject *)PyArray_EMPTY(2, dims, NPY_COMPLEX64, 0);
    int succeeded = multislice_propagate_fftw((fftwf_complex *)wave_in->data, (int)dims[0], (int)dims[1],
                                              (float *)slices->data, n_slices,  pixel_size, dz,
                                              wave_length, relativity_gamma,
                                              (fftwf_complex *)wave_out->data);
    if (!succeeded) {
        Py_RETURN_NONE;
    }
    return PyArray_Return(wave_out);
}


/* Method table, Module definition and Module initialization function */
static PyMethodDef em_kernel_methods[] = {
    {
        "multislice_propagate_fftw", (PyCFunction)multislice_propagate_fftw_wrapper, METH_VARARGS | METH_KEYWORDS,
        ""
    },
    {NULL, NULL, 0, NULL}
};


static PyModuleDef em_kernel = {
    PyModuleDef_HEAD_INIT,
    "em_kernel",
    "This module provides core functions for multi-slice em imaging simulation.",
    -1,
    em_kernel_methods
};

PyMODINIT_FUNC PyInit_em_kernel(void)
{
    PyObject *module;
    module = PyModule_Create(&em_kernel);
    import_array();
    // import_ufunc();
    return module;
}