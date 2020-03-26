# Copyright (c) 2017-2020, Lawrence Livermore National Security, LLC and
# other Shroud Project Developers.
# See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
# 
########################################################################
"""
Generate a module for tutorial using PyBindGen
"""

import pybindgen
from pybindgen import (param, retval)

def generate(fp):
    mod = pybindgen.Module('tutorial')
    mod.add_include('"tutorial.hpp"')
    namespace = mod.add_cpp_namespace('tutorial')

    namespace.add_enum('Color', ['RED', 'BLUE', 'WHITE'])
#    mod.add_function('AcceptEnum', None, [param('MyEnum_e', 'value')])

    # default arguments
    namespace.add_function(
        'UseDefaultArguments', 'double',
        [param('double', 'arg1', default_value='3.1415'),
         param('bool', 'arg2', default_value='true')])

    # overloaded
    namespace.add_function(
        'OverloadedFunction', None, 
        [param('const std::string &', 'name')])
    namespace.add_function(
        'OverloadedFunction', None, 
        [param('int', 'index')])

    # overloaded with default arguments
    namespace.add_function(
        'UseDefaultOverload', 'int',
        [param('int', 'num'),
         param('int', 'offset', default_value='0'),
         param('int', 'stride', default_value='1')])
    namespace.add_function(
        'UseDefaultOverload', 'int',
        [param('double', 'type'),
         param('int', 'num'),
         param('int', 'offset', default_value='0'),
         param('int', 'stride', default_value='1')])

    mod.generate(fp)

if __name__ == '__main__':
    import sys
    generate(sys.stdout)
