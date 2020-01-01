# Copyright (c) 2017-2020, Lawrence Livermore National Security, LLC and
# other Shroud Project Developers.
# See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

from distutils.core import setup, Extension

extension_mod = Extension(
    "_cstruct",
    ["swigstruct_module.c", "../struct.c"],
    include_dirs=['..'],
)

setup(name = "cstruct", ext_modules=[extension_mod])
