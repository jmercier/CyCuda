#!/usr/bin/python -B
from config import *

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext


extensions = [ Extension("cycuda.core", ['src/core.pyx'],
                         libraries = [cuda_library],
                         library_dirs = [cuda_library_dir],
                         include_dirs = [cuda_include_dir, 'src'])]
"""
Extension("cycuda.garray", ['src/garray.pyx'],
libraries = [cuda_library],
library_dirs = [cuda_library_dir],
include_dirs = [cuda_include_dir, 'src'])
"""

cycuda_srcs = ['cycuda']

setup(
    cmdclass = {'build_ext': build_ext},
    ext_modules = extensions,
    packages = ['cycuda'],
    name = 'CyCuda',
    version = '0.0.1',
    description = "Cython Bindings for Cuda",
    author = "J-Pascal Mercier",
    author_email = "jp.mercier@gmail.com",
    license = "MIT"
)



