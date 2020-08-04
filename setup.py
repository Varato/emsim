from setuptools import setup, find_packages, Extension
import platform
import numpy as np

ext_modules = []
data_files = []

if platform.system() == "Windows" and platform.machine() == "AMD64":
    internal_lib_dir = "lib_x86_64-win32"
    internal_include_dir = "emsim/ext/include"
    dlls = ["libfftw3f-3.dll"]
    data_files.append(("emsim/ext", [internal_lib_dir + '/' + dll for dll in dlls]))

    dens_kernel = Extension(
        "emsim.ext.dens_kernel",
        sources=[
            "emsim/ext/src/dens_kernel.c",
            "emsim/ext/src/dens_kernel_pymodule.c"],
        include_dirs=[internal_include_dir, np.get_include()],
        library_dirs=[internal_lib_dir],
        libraries=['libfftw3f-3'],  # using single float fftw: change fftw_ to fftwf_ in c files
        extra_compile_args=['/openmp'],
    )
    ext_modules.append(dens_kernel)

    em_kernel = Extension(
        "emsim.ext.em_kernel",
        sources=[
            "emsim/ext/src/em_kernel.c",
            "emsim/ext/src/em_kernel_pymodule.c"],
        include_dirs=[internal_include_dir, np.get_include()],
        library_dirs=[internal_lib_dir],
        libraries=['libfftw3f-3'],  # using single float fftw: change fftw_ to fftwf_ in c files
        extra_compile_args=['/openmp'],
    )
    ext_modules.append(em_kernel)


setup(
    name="emsim",
    version="0.0.0",
    ext_modules = ext_modules,
    packages=find_packages(where='.'),
    scripts=[],
    install_requires=[],
    package_data = {
        'atom_params': ['emsim/assets/*.txt'],
        'cuda_kernels': ['emsim/cuda/kernels/*.cu']
    },
    data_files = data_files
)
