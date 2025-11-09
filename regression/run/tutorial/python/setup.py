#!/usr/bin/env python
# Copyright Shroud Project Developers. See LICENSE file for details.
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
config = shroud.create_wrapper('../../../tutorial.yaml',
                               path=['../../..'],
                               outdir=outdir)

tutorial = Extension(
    'tutorial',
    sources = config.pyfiles + ['../tutorial.cpp'],
    include_dirs=[numpy.get_include(), '..']
)

setup(
    name='tutorial',
    version="0.0",
    description='shroud tutorial',
    author='xxx',
    author_email='yyy@zz',
    ext_modules=[tutorial],
)

