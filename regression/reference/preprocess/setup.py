# setup.py
# This file is generated by Shroud nowrite-version. Do not edit.
# Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and
# other Shroud Project Developers.
# See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
#
from setuptools import setup, Extension

module = Extension(
    'preprocess',
    sources=[
         'pyUser1type.cpp',
         'pyUser2type.cpp',
         'pypreprocessmodule.cpp',
         'pypreprocessutil.cpp'
    ],
    language='c++',
    include_dirs = None,
#    libraries = ['tcl83'],
#    library_dirs = ['/usr/local/lib'],      
#    extra_compile_args = [ '-O0', '-g' ],
#    extra_link_args =
)

setup(
    name='preprocess',
    ext_modules = [module],
)
