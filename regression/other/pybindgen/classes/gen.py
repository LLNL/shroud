# Copyright (c) 2017-2020, Lawrence Livermore National Security, LLC and
# other Shroud Project Developers.
# See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
# 
########################################################################
"""
Generate a module for classes using PyBindGen
"""

import pybindgen
from pybindgen import (param, retval)

def generate(fp):
    mod = pybindgen.Module('classes')
    mod.add_include('"classes.hpp"')
    namespace = mod.add_cpp_namespace('classes')

    class1 = namespace.add_class('Class1')
    class1.add_enum('DIRECTION', ['UP', 'DOWN', 'LEFT', 'RIGHT'])
#    class1.add_function('AcceptEnum', None, [param('MyEnum_e', 'value')])

    class1.add_instance_attribute('m_flag', 'int')
    class1.add_constructor([param('int', 'flag')])
    class1.add_constructor([])
    class1.add_method('Method1', None, [])

    sclass = namespace.add_class("Singleton", is_singleton=True)
    sclass.add_method("getReference", retval("classes::Singleton&", caller_owns_return=True), [],
                      is_static=True)

#    mod.add_class('Class1',
#                  memory_policy=cppclass.ReferenceCountingMethodsPolicy(
#                      incref_method='Ref',
#                      decref_method='Unref',
#                      peekref_method='PeekRef')
#    )
#    mod.add_function('DoSomething', retval('Class1 *', caller_owns_return=False), [])

    mod.generate(fp)

if __name__ == '__main__':
    import sys
    generate(sys.stdout)
