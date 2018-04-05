# Copyright (c) 2017-2018, Lawrence Livermore National Security, LLC. 
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
# #######################################################################
#
# Shroud tests
#
# (from parent directory) python -m unittest test
#
from __future__ import absolute_import

import unittest

from . import test_ast
from . import test_declast
from . import test_generate
#from . import test_typemap
from . import test_util
from . import test_wrapf
from . import test_wrapp
#from . import test_shroud


test_cases = (
    test_util.UtilCase,
    test_util.ScopeCase,
    test_ast.Namespace,
    test_declast.CheckParse,
    test_declast.CheckExpr,
    test_declast.CheckNamespace,
    test_declast.CheckTypedef,
    test_declast.CheckEnum,
    test_declast.CheckClass,
    test_ast.CheckAst,
    test_generate.CheckImplied,
    test_wrapf.CheckAllocatable,
    test_wrapp.CheckImplied,
#    test_shroud.MainCase,
)


def load_tests(loader, tests, pattern):
    # used from 'python -m unittest tests'
    suite = unittest.TestSuite()
    for test_class in test_cases:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    return suite


def load_tests2():
    # used from 'setup.py test'
    loader = unittest.TestLoader()
    return load_tests(loader, None, None)

