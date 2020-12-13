from setuptools import setup
from setuptools import find_packages


package_data = {"emsim.assets": ["atom_mass.txt", "atom_params.txt"]}

setup(
    name='emsim',
    version='0.1.1',
    author='Varato',
    author_email='imxinchen@outlook.com',
    description='an electron microscopy imaging simulator',
    long_description='',
    packages=find_packages(where='.'),
    package_data=package_data,
    zip_safe=False,
)
