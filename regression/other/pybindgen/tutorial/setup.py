#!/usr/bin/env python
# Copyright (c) 2017-2020, Lawrence Livermore National Security, LLC and
# other Shroud Project Developers.
# See the top-level COPYRIGHT file for details.
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

