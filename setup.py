from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    ext_modules = cythonize(["./DataStructures/point.pyx", "./DataStructures/decomposition.pyx", "utils.pyx", "./DataStructures/myheap.pyx", "./Algorithms/hungarian.pyx", "./Algorithms/pWasserstein.pyx"]),
    include_dirs=[numpy.get_include()],
    compiler_directives={'profile': True}
)