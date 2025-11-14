# Copyright Shroud Project Developers. See LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

"""
Shroud tests

 (from parent directory) python -m unittest test
"""

import unittest

from . import (test_ast, test_declast, test_format, test_generate,
               test_statements, test_util, test_wrapf, test_wrapp)

# from . import test_shroud


test_cases = (
    test_util.UtilCase,
    test_util.ScopeCase,
    test_statements.Statements,
    test_ast.Namespace,
    test_declast.CheckParse,
    test_declast.CheckExpr,
    test_declast.CheckNamespace,
    test_declast.CheckTypedef,
    test_declast.CheckEnum,
    test_declast.CheckStruct,
    test_declast.CheckClass,
    test_ast.CheckAst,
    test_format.WFormat,
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
