//
// Created by Chen on 11/8/2020.
//
#include <cstdio>

int maxThreadsPerblock;

void cudaInit() {
    cudaDeviceProp prop{};
    if(cudaGetDeviceProperties (&prop, 0) != cudaSuccess) {
        fprintf(stderr, "CUDA init error: %s\n", cudaGetErrorString(cudaGetLastError()));
        return;
    }
    maxThreadsPerblock = prop.maxThreadsPerBlock;
}