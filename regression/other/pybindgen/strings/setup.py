# Copyright (c) 2017-2020, Lawrence Livermore National Security, LLC and
# other Shroud Project Developers.
# See the top-level COPYRIGHT file for details.
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
