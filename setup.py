from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize
import numpy
import platform

ext_options = {"compiler_directives": {"language_level": 3}}

compile_args = ['-fopenmp']
link_args = ['-fopenmp']

if platform.system() == "Darwin":
    compile_args = ['-Xpreprocessor', '-fopenmp']
    link_args = ['-lomp', '-Wno-everything']

ext_modules = cythonize([
    Extension("*", ["giant/ray_tracer/*.pyx"], extra_compile_args=compile_args,
              extra_link_args=link_args, include_dirs=[numpy.get_include()]),
    Extension("*", ["giant/ray_tracer/shapes/*.pyx"], extra_compile_args=compile_args,
              extra_link_args=link_args, include_dirs=[numpy.get_include()]),
    Extension("*", ["giant/relative_opnav/estimators/sfn/sfn_correlators.pyx"],
              extra_compile_args=compile_args, extra_link_args=link_args,
              include_dirs=[numpy.get_include()])
], **ext_options)

setup(ext_modules=ext_modules)