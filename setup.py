# Copyright 2021 United States Government as represented by the Administrator of the National Aeronautics and Space
# Administration.  No copyright is claimed in the United States under Title 17, U.S. Code. All Other Rights Reserved.

from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize

from sys import platform

import numpy

ext_options = {"compiler_directives": {"language_level": 3}}
# ext_options = {"compiler_directives": {"language_level": 2}}

if platform.lower().startswith('win32'):
    ext_modules = [
        Extension("*",
                  ["giant/ray_tracer/*.pyx"],
                  #extra_compile_args=['/openmp:llvm'],
                  include_dirs=[numpy.get_include()]
                  ),
        Extension("*",
                  ["giant/ray_tracer/shapes/*.pyx"],
                  #extra_compile_args=['/openmp:llvm'],
                  include_dirs=[numpy.get_include()]
                  ),
        Extension("*",
                  ["giant/relative_opnav/estimators/sfn/sfn_correlators.pyx"],
                  #extra_compile_args=['/openmp:llvm'],
                  include_dirs=[numpy.get_include()]
                  )
    ]  # untested...
elif "darwin" in platform.lower():
    ext_modules = [
        Extension("*",
                  ["giant/ray_tracer/*.pyx"],
                  extra_compile_args=['-Xpreprocessor', '-fopenmp'],
                  extra_link_args=['-lomp', '-Wno-everything'],
                  include_dirs=[numpy.get_include()],

                  ),
        Extension("*",
                  ["giant/ray_tracer/shapes/*.pyx"],
                  extra_compile_args=['-Xpreprocessor', '-fopenmp'],
                  extra_link_args=['-lomp', '-Wno-everything'],
                  include_dirs=[numpy.get_include()],
                  ),
        Extension("*",
                  ["giant/relative_opnav/estimators/sfn/sfn_correlators.pyx"],
                  extra_compile_args=['-Xpreprocessor', '-fopenmp'],
                  extra_link_args=['-lomp', '-Wno-everything'],
                  include_dirs=[numpy.get_include()],
                  )
    ]
else:
    ext_modules = [
        Extension("*",
                  ["giant/ray_tracer/*.pyx"],
                  extra_compile_args=['-fopenmp'],
                  extra_link_args=['-fopenmp'],
                  include_dirs=[numpy.get_include()],

                  ),
        Extension("*",
                  ["giant/ray_tracer/shapes/*.pyx"],
                  extra_compile_args=['-fopenmp'],
                  extra_link_args=['-fopenmp'],
                  include_dirs=[numpy.get_include()],

                  ),
        Extension("*",
                  ["giant/relative_opnav/estimators/sfn/sfn_correlators.pyx"],
                  extra_compile_args=['-fopenmp'],
                  extra_link_args=['-fopenmp'],
                  include_dirs=[numpy.get_include()],
                  )
    ]

setup(name='giant',
      version='1.0',
      description="Goddard Image Analysis and Navigation Tool",
      long_description="Copyright 2021 United States Government as represented by the Administrator of the National "
                       "Aeronautics and Space Administration.  No copyright is claimed in the United States under "
                       "Title 17, U.S. Code. All Other Rights Reserved.",
      author='Andrew Liounis',
      author_email='andrew.j.liounis@nasa.gov',
      license='NOSA 1.3',
      packages=['giant', 'giant.calibration', 'giant.catalogues', 'giant.ray_tracer', 'giant.ray_tracer.shapes', 'giant.point_spread_functions',
                'giant.relative_opnav', 'giant.relative_opnav.estimators', 'giant.relative_opnav.estimators.sfn', 
                'giant.stellar_opnav', "giant.utilities", "giant.scripts", "giant.camera_models", "giant.ufo"],
      ext_modules=cythonize(ext_modules, **ext_options),
      install_requires=[
          'pandas',
          'numpy',
          'astropy',
          'scipy',
          'matplotlib',
          'lxml',
          'cython',
          'python-dateutil',
          'psutil',
          'spiceypy',
          'dill',
          'astroquery'
      ],
      package_dir={'giant.catalogues': 'giant/catalogues'},
      package_data={'giant.catalogues': ['data/*']},
      entry_points={
          "console_scripts": [
              "ingest_shape = giant.scripts.ingest_shape:main",
              "build_catalogue = giant.scripts.build_catalogue:main",
              "generate_sample_data = giant.scripts.generate_sample_data:main",
              "merge_cameras = giant.scripts.merge_cameras:main",
              "shape_stats = giant.scripts.shape_stats:main",
              "spc_to_results = giant.scripts.spc_to_results:main",
              "spc_to_feature_catalogue = giant.scripts.spc_to_feature_catalogue:main",
              "tile_shape = giant.scripts.tile_shape:main",
          ]
      },
      zip_safe=False)
