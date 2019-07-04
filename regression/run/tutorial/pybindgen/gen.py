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
        'Function5', 'double',
        [param('double', 'arg1', default_value='3.1415'),
         param('bool', 'arg2', default_value='true')])

    # overloaded
    namespace.add_function(
        'Function6', None, 
        [param('const std::string &', 'name')])
    namespace.add_function(
        'Function6', None, 
        [param('int', 'index')])

    # overloaded with default arguments
    namespace.add_function(
        'overload1', 'int',
        [param('int', 'num'),
         param('int', 'offset', default_value='0'),
         param('int', 'stride', default_value='1')])
    namespace.add_function(
        'overload1', 'int',
        [param('double', 'type'),
         param('int', 'num'),
         param('int', 'offset', default_value='0'),
         param('int', 'stride', default_value='1')])

    class1 = namespace.add_class('Class1')
    class1.add_enum('DIRECTION', ['UP', 'DOWN', 'LEFT', 'RIGHT'])
#    class1.add_function('AcceptEnum', None, [param('MyEnum_e', 'value')])

    class1.add_instance_attribute('m_flag', 'int')
    class1.add_constructor([param('int', 'flag')])
    class1.add_constructor([])
    class1.add_method('Method1', None, [])

    struct = namespace.add_struct('struct1')
    struct.add_instance_attribute('i', 'int')
    struct.add_instance_attribute('d', 'double')

    sclass = mod.add_class("Singleton", is_singleton=True)
    sclass.add_method("instancePtr", retval("Singleton*", caller_owns_return=True), [],
                      is_static=True)

    mod.generate(fp)

if __name__ == '__main__':
    import sys
    generate(sys.stdout)
