# Copyright Shroud Project Developers. See LICENSE file for details.

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

