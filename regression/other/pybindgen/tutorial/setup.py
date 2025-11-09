#!/usr/bin/env python
# Copyright Shroud Project Developers. See LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
########################################################################

from distutils.core import setup, Extension

tutorial = Extension(
    'tutorial',
    sources = ['tutorial-binding.cpp', 'tutorial.cpp'],
)

setup(
    name='PyBindGen-tutorial',
    description='PyBindGen tutorial',
    ext_modules=[tutorial],
)

