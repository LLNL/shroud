# Copyright Shroud Project Developers. See LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

from distutils.core import setup, Extension
from Cython.Build import cythonize

setup(
    name = 'Test structures',
#    ext_modules = cythonize(
#        "cstruct.pyx",
#        include_path=[".."],
#    ),
    ext_modules=cythonize(
        Extension(
            "cstruct",
            sources=["cstruct.pyx", "struct.c"],
            extra_compile_args = ["-std=c99"],
        ),
        annotate=True,
#        language='c++',
    ),
)

