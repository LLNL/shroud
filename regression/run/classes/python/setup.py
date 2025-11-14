#!/usr/bin/env python
# Copyright Shroud Project Developers. See LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
########################################################################

import os
from distutils.core import Extension, setup

import numpy

import shroud

outdir = 'build/source'
if not os.path.exists(outdir):
    os.makedirs(outdir)
config = shroud.create_wrapper('../../../classes.yaml',
                               path=['../../..'],
                               outdir=outdir)

classes = Extension(
    'classes',
    sources = config.pyfiles + ['../classes.cpp'],
    include_dirs=[numpy.get_include(), '..']
)

setup(
    name='classes',
    version="0.0",
    description='shroud classes',
    author='xxx',
    author_email='yyy@zz',
    ext_modules=[classes],
)
