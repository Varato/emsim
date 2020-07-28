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
    int len_x = (int) slices->dimensions[1];
    int len_y = (int) slices->dimensions[2];
    printf("wave shape %d, %d\n", wave_in->dimensions[0], wave_in->dimensions[1]);

    float *wave_out;
    wave_out = fftwf_malloc(sizeof(fftwf_complex) * len_x * len_y);
    if (!wave_out) {
        Py_RETURN_NONE;
    }

    int succeeded = multislice_propagate_fftw((fftwf_complex *)wave_in->data, len_x, len_y,
                                              (float *)slices->data, n_slices,  pixel_size, dz,
                                              wave_length, relativity_gamma,
                                              wave_out);
    if (!succeeded) {
        Py_RETURN_NONE;
    }

    npy_intp dims[2] = {len_x, len_y};
    
    PyArrayObject *ret; // return value
    ret = (PyArrayObject *)PyArray_SimpleNewFromData(2, dims, NPY_COMPLEX64, wave_out);
    return PyArray_Return(ret);
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