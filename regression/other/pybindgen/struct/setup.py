# Copyright (c) 2017-2020, Lawrence Livermore National Security, LLC and
# other Shroud Project Developers.
# See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
########################################################################

from distutils.core import setup, Extension

struct = Extension(
    'struct',
    sources = ['struct-binding.c', 'struct.c'],
)

setup(
    name='PyBindGen-struct',
    description='PyBindGen struct',
    ext_modules=[struct],
)

