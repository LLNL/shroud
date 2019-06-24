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

#    mod.add_class('Class1',
#                  memory_policy=cppclass.ReferenceCountingMethodsPolicy(
#                      incref_method='Ref',
#                      decref_method='Unref',
#                      peekref_method='PeekRef')
#    )
#    mod.add_function('DoSomething', retval('Class1 *', caller_owns_return=False), [])

    mod.generate(fp)
