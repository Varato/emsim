from setuptools import setup, find_packages, Extension
import numpy as np

setup(
    name="emsim",
    version="0.0.0",
    packages=find_packages(where='.'),
    scripts=[],
    install_requires=[],
    package_data={
        'atom_params': ['emsim/assets/*.txt'],
    },
)
