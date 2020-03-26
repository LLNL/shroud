# Copyright (c) 2017-2020, Lawrence Livermore National Security, LLC and
# other Shroud Project Developers.
# See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
########################################################################
"""
Generate a module for strings using PyBindGen
"""

import pybindgen
from pybindgen import (cppclass, Parameter, param, retval)

def generate(fp):
    mod = pybindgen.Module('strings')
    mod.add_include('"strings.hpp"')
    mod.add_function('passChar', None, 
                     [pybindgen.param('char', 'status')])
    mod.add_function('returnChar', retval('char'), [])
    mod.add_function('returnStrings', None,
                     [param('std::string &', 'arg1', direction=Parameter.DIRECTION_OUT),
                      param('std::string &', 'arg2', direction=Parameter.DIRECTION_OUT)])

    mod.add_function('getConstStringAlloc',
                     retval('const std::string'), [])
    mod.add_function('getConstStringRefAlloc',
                     retval('const std::string &'), [])
#    mod.add_function('getConstStringPtrAlloc',
#                     retval('const std::string *'), [])
#pybindgen.typehandlers.base.TypeLookupError: ['std::string *']

    mod.generate(fp)
