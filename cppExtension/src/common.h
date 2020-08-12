//
// Created by Chen on 12/8/2020.
//

#ifndef EMSIM_COMMON_H
#define EMSIM_COMMON_H

#include <pybind11/numpy.h>

namespace py = pybind11;

/*
 * defined in buffer_info.h
 *
 * struct buffer_info {
 *     void *ptr;           // if nullptr, numpy will allocate the memory
 *     ssize_t itemsize;
 *     std::string format;  // should be set by py::format_descriptor<T>::format()
 *     ssize_t ndim;
 *     std::vector<ssize_t> shape;
 *     std::vector<ssize_t> strides;
 * };
 * */

template<typename T>
py::array_t<T> make2dArray(size_t n1, size_t n2) {
    // construct row majoc (c contiguous) numpy array
    py::buffer_info info (
            nullptr,
            sizeof(T),
            py::format_descriptor<T>::format(),
            2,
            {n1, n2},
            {sizeof(T)*n2, sizeof(T)});
    return py::array_t<T>(info);
}

template<typename T>
py::array_t<T> make3dArray(size_t n0, size_t n1, size_t n2) {
    // construct row majoc (c contiguous) numpy array
    py::buffer_info info (
            nullptr,
            sizeof(T),
            py::format_descriptor<T>::format(),
            3,
            {n0, n1, n2},
            {sizeof(T)*n1*n2, sizeof(T)*n2, sizeof(T)});
    return py::array_t<T>(info);
}

#endif //EMSIM_COMMON_H
