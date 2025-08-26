# Copyright Shroud Project Developers. See LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

from distutils.core import setup, Extension

strings = Extension(
    'strings',
    sources = ['strings-binding.cpp', 'strings.cpp'],
)

setup(
    name='PyBindGen-strings',
    description='PyBindGen strings',
    ext_modules=[strings],
)
