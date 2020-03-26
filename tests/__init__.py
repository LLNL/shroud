# Copyright (c) 2017-2020, Lawrence Livermore National Security, LLC and
# other Shroud Project Developers.
# See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

"""
Shroud tests

 (from parent directory) python -m unittest test
"""

from __future__ import absolute_import

import unittest

from . import test_ast
from . import test_declast
from . import test_generate
from . import test_typemap
from . import test_util
from . import test_wrapf
from . import test_wrapp

# from . import test_shroud


test_cases = (
    test_util.UtilCase,
    test_util.ScopeCase,
    test_typemap.Typemap,
    test_ast.Namespace,
    test_declast.CheckParse,
    test_declast.CheckExpr,
    test_declast.CheckNamespace,
    test_declast.CheckTypedef,
    test_declast.CheckEnum,
    test_declast.CheckStruct,
    test_declast.CheckClass,
    test_ast.CheckAst,
    test_generate.CheckImplied,
    test_wrapf.CheckAllocatable,
    test_wrapp.CheckImplied,
    test_wrapp.CheckStruct,
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
