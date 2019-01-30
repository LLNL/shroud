#!/usr/bin/env python
# Copyright (c) 2018-2019, Lawrence Livermore National Security, LLC. 
#
# Produced at the Lawrence Livermore National Laboratory 
# 
# LLNL-CODE-738041.
#
# All rights reserved. 
#  
# This file is part of Shroud.
#
# For details about use and distribution, please read LICENSE.
# 
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

