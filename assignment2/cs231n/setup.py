#from distutils.core import setup
#from distutils.extension import Extension
#the problem that Unable to find vcvarsall.bat is solved by the solution on https://github.com/cython/cython/wiki/CythonExtensionsOnWindows

from setuptools import setup
from setuptools import Extension
from Cython.Build import cythonize
import numpy

extensions = [
  Extension('im2col_cython', ['im2col_cython.pyx'],
    include_dirs = [numpy.get_include()]),
]

setup(
    ext_modules = cythonize(extensions),
)
