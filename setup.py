from setuptools import setup, find_packages, Extension
import numpy as np

dens_kernel = Extension(
    "emsim.extensions.dens_kernel", 
    sources = [
        "emsim/extensions/src/dens_kernel_pymodule.c",
        "emsim/extensions/src/dens_kernel.c",
        "emsim/extensions/src/utils.c"],
    include_dirs = ["emsim/extensions/include", np.get_include()],
    library_dirs = ["emsim/extensions/lib/win-x64"],
    libraries = ['libfftw3-3', 'python38']
)

setup(
    name="emsim",
    version="0.0.0",
    ext_modules = [dens_kernel],
    packages=find_packages(where='.'),
    scripts=[],
    install_requires=[],
    package_data={
        'atom_params': ['emsim/assets/*.txt'],
    },
)
