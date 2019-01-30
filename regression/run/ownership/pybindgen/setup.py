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
from gen import generate

try:
    os.mkdir("build")
except OSError:
    pass
module_fname = "ownership-binding.cpp"
with open(module_fname, "wt") as fp:
    print("Generating file {}".format(module_fname))
    generate(fp)

ownership = Extension(
    'ownership',
    sources = [module_fname, '../ownership.cpp'],
    include_dirs=['..']
)

setup(
    name='PyBindGen-ownership',
    version="0.0",
    description='PyBindGen ownership',
    author='xxx',
    author_email='yyy@zz',
    ext_modules=[ownership],
)

