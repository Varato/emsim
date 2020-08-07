from skbuild import setup
from setuptools import find_packages

setup(
    name="emsim",
    version="0.0.0",
    # ext_modules=ext_modules,
    packages=find_packages(where='.'),
    cmake_args=['-DCMAKE_BUILD_TYPE=Release', '-DWITH_CUDA_EXT=ON'],
    scripts=[],
    install_requires=[],
    package_data={
        'atom_params': ['emsim/assets/*.txt'],
        'cuda_kernels': ['emsim/cuda/kernels/*.cu']
    },
)
