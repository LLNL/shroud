# Copyright (c) 2017-2019, Lawrence Livermore National Security, LLC and
# other Shroud Project Developers.
# See the top-level COPYRIGHT file for details.

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

