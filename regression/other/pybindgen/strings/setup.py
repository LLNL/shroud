# Copyright (c) 2017-2020, Lawrence Livermore National Security, LLC and
# other Shroud Project Developers.
# See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

import os
from distutils.core import setup, Extension
from gen import generate

try:
    os.mkdir("build")
except OSError:
    pass
module_fname = "strings-binding.cpp"
with open(module_fname, "wt") as fp:
    print("Generating file {}".format(module_fname))
    generate(fp)

strings = Extension(
    'strings',
    sources = [module_fname, '../strings.cpp'],
    include_dirs=['..']
)

setup(
    name='PyBindGen-strings',
    version="0.0",
    description='PyBindGen strings',
    author='xxx',
    author_email='yyy@zz',
    ext_modules=[strings],
)
