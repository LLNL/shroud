#!/usr/bin/env python
# Copyright (c) 2017-2020, Lawrence Livermore National Security, LLC and
# other Shroud Project Developers.
# See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
########################################################################

import os
from distutils.core import setup, Extension
from gen import generate

try:
    os.mkdir("build")
except OSError:
    pass
module_fname = "tutorial-binding.cpp"
with open(module_fname, "wt") as fp:
    print("Generating file {}".format(module_fname))
    generate(fp)

tutorial = Extension(
    'tutorial',
    sources = [module_fname, '../../run-tutorial/tutorial.cpp'],
    include_dirs=['../../run-tutorial']
)

setup(
    name='PyBindGen-tutorial',
    version="0.0",
    description='PyBindGen tutorial',
    author='xxx',
    author_email='yyy@zz',
    ext_modules=[tutorial],
)

