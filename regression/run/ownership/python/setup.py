#!/usr/bin/env python
# Copyright (c) 2017-2020, Lawrence Livermore National Security, LLC and
# other Shroud Project Developers.
# See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
########################################################################

import os
from distutils.core import setup, Extension
import shroud
import numpy

outdir = 'build/source'
if not os.path.exists(outdir):
    os.makedirs(outdir)
config = shroud.create_wrapper('../../../ownership.yaml', outdir=outdir)

ownership = Extension(
    'ownership',
    sources = config.pyfiles + ['../ownership.cpp'],
    include_dirs=[numpy.get_include(), '..']
)

setup(
    name='ownership',
    version="0.0",
    description='shroud ownership',
    author='xxx',
    author_email='yyy@zz',
    ext_modules=[ownership],
)

