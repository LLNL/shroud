import pybindgen
from pybindgen import (cppclass, Module, Parameter, ReturnValue)
def generate(fp):
    mod = Module('foo')
    mod.add_include ('"ownership.hpp"')

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

    mod.generate(fp)

if __name__ == '__main__':
    import sys
    generate(sys.stdout)
