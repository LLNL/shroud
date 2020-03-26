# Copyright (c) 2017-2020, Lawrence Livermore National Security, LLC and
# other Shroud Project Developers.
# See the top-level COPYRIGHT file for details.

from distutils.core import setup, Extension

ownership = Extension(
    'ownership',
    sources = ['ownership-binding.cpp', 'ownership.cpp'],
)

setup(
    name='PyBindGen-ownership',
    description='PyBindGen ownership',
    ext_modules=[ownership],
)

