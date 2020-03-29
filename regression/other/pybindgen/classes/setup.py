# Copyright (c) 2017-2020, Lawrence Livermore National Security, LLC and
# other Shroud Project Developers.
# See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
########################################################################

from distutils.core import setup, Extension

classes = Extension(
    'classes',
    sources = ['classes-binding.cpp', 'classes.cpp'],
)

setup(
    name='PyBindGen-classes',
    description='PyBindGen classes',
    ext_modules=[classes],
)

