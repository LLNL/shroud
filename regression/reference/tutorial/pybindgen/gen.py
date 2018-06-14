# Copyright (c) 2018, Lawrence Livermore National Security, LLC. 
# Produced at the Lawrence Livermore National Laboratory 
# 
# LLNL-CODE-738041.
# All rights reserved. 
#  
# This file is part of Shroud.  For details, see
# https://github.com/LLNL/shroud. Please also read shroud/LICENSE.
#  
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#  
# * Redistributions of source code must retain the above copyright
#   notice, this list of conditions and the disclaimer below.
# 
# * Redistributions in binary form must reproduce the above copyright
#   notice, this list of conditions and the disclaimer (as noted below)
#   in the documentation and/or other materials provided with the
#   distribution.
# 
# * Neither the name of the LLNS/LLNL nor the names of its contributors
#   may be used to endorse or promote products derived from this
#   software without specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL LAWRENCE
# LIVERMORE NATIONAL SECURITY, LLC, THE U.S. DEPARTMENT OF ENERGY OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
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
