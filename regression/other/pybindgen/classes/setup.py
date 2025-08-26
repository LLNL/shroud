# Copyright Shroud Project Developers. See LICENSE file for details.
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

