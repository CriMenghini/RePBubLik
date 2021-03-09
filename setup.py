import numpy
from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize


extensions = [
    Extension("random_walks", ["random_walks_all.pyx"], include_dirs=[numpy.get_include()])
]

setup(
    name='random_walks',
    ext_modules=cythonize(extensions),
    zip_safe=False,
    script_args = ["build_ext", "--inplace"]
)