# Copyright (c) 2017-2020, Lawrence Livermore National Security, LLC and
# other Shroud Project Developers.
# See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
#
########################################################################
"""
Generate a module for ownership using PyBindGen
"""

import pybindgen
from pybindgen import (cppclass, Module, param, Parameter, ReturnValue)
def generate(fp):
    mod = Module('foo')
    mod.add_include ('"ownership.hpp"')

    Foo = mod.add_class('Foo', automatic_type_narrowing=True)
    Foo.add_constructor([Parameter.new('std::string', 'datum')])
    Foo.add_constructor([])
    Foo.add_constructor([Parameter.new('const Foo&', 'foo')])

    ## Zbr is a reference counted class
    Zbr = mod.add_class('Zbr',
                        memory_policy=cppclass.ReferenceCountingMethodsPolicy(
                            incref_method='Ref',
                            decref_method='Unref',
                            peekref_method="GetReferenceCount"))
#                        allow_subclassing=True)

    Zbr.add_constructor([])
    Zbr.add_constructor([Parameter.new('std::string', 'datum')])
    Zbr.add_method('get_datum', ReturnValue.new('std::string'), [])
    Zbr.add_method('get_int', ReturnValue.new('int'), [Parameter.new('int', 'x')],
                             is_virtual=True)
    Zbr.add_static_attribute('instance_count', ReturnValue.new('int'))
    Zbr.add_method('get_value', ReturnValue.new('int'),
                   [Parameter.new('int*', 'x', direction=Parameter.DIRECTION_OUT)])

    mod.add_function('store_zbr', None,
                     [Parameter.new('Zbr*', 'zbr', transfer_ownership=True)])
    mod.add_function('invoke_zbr', ReturnValue.new('int'), [Parameter.new('int', 'x')])
    mod.add_function('delete_stored_zbr', None, [])

    SomeObject = mod.add_class('SomeObject', allow_subclassing=True)

    SomeObject.add_method('set_foo_ptr', ReturnValue.new('void'),
                          [Parameter.new('Foo*', 'foo', transfer_ownership=True)])
    SomeObject.add_method('set_foo_shared_ptr', ReturnValue.new('void'),
                          [Parameter.new('Foo*', 'foo', transfer_ownership=False)])

    SomeObject.add_method('get_foo_shared_ptr', ReturnValue.new('const Foo*', caller_owns_return=False), [])
    SomeObject.add_method('get_foo_ptr', ReturnValue.new('Foo*', caller_owns_return=True), [])



    mod.generate(fp)

if __name__ == '__main__':
    import sys
    generate(sys.stdout)
