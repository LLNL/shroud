# Copyright (c) 2017-2020, Lawrence Livermore National Security, LLC and
# other Shroud Project Developers.
# See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
# 
########################################################################
"""
Generate a module for struct using PyBindGen
"""

import pybindgen
from pybindgen import (param, retval)

def generate(fp):
    mod = pybindgen.Module('cstruct')
    mod.add_include('"struct.h"')

    struct1 = mod.add_struct('Cstruct1')
    struct1.add_instance_attribute('ifield', 'int')
    struct1.add_instance_attribute('dfield', 'double')

    mod.generate(fp)

if __name__ == '__main__':
    import sys
    generate(sys.stdout)
