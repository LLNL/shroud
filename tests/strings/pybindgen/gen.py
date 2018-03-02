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
#                     retval('const std::string &', is_const=True), [])
#    mod.add_function('getConstStringPtrAlloc',
#                     retval('std::string *'), [])
#pybindgen.typehandlers.base.TypeLookupError: ['std::string *']

#    mod.add_class('Class1',
#                  memory_policy=cppclass.ReferenceCountingMethodsPolicy(
#                      incref_method='Ref',
#                      decref_method='Unref',
#                      peekref_method='PeekRef')
#    )
#    mod.add_function('DoSomething', retval('Class1 *', caller_owns_return=False), [])

    mod.generate(fp)
