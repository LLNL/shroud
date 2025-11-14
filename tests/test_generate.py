# Copyright Shroud Project Developers. See LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

import unittest

from shroud import ast, error, generate

ShroudParseError = error.ShroudParseError

class Cursor(error.Cursor):
    """Mock class for error.Cursor
    Record last error message.
    """
    def push_phase(self, name):
        pass
    def pop_phase(self, name):
        pass
    def push_node(self, node):
        pass
    def pop_node(self, node):
        pass
    def generate(self, message):
        self.message = message
        
error.cursor = Cursor()

class Config(object):
    def __init__(self):
        pass


class CheckImplied(unittest.TestCase):
    def setUp(self):
        # Create a dictionary of parsed arguments
        self.library = ast.LibraryNode()
        node = self.library.add_function(
            "void func1("
            "int *array  +intent(in)+dimension(:),"
            "int *array2 +intent(in)+dimension(:,:),"
            "int  scalar +intent(in)+implied(size(array))"
            ")"
        )
        self.func1 = node

    def XXXtest_dimension_1(self):
        # XXX - moved to FillMeta - checked with error-generate.yaml
        # Check missing dimension value
        # (:) used to be accepted as assumed shape -- now rank(1).
        library = ast.LibraryNode()
        node = self.library.add_function(
            "void func1(const int *array  +dimension)"
        )
        config = Config()
        vfy = generate.VerifyAttrs(library, config)

        vfy.check_fcn_attrs(node)
        self.assertTrue("dimension attribute must have a value" in str(error.cursor.message))

    def XXXtest_dimension_2(self):
        # XXX - moved to FillMeta - checked with error-generate.yaml
        # Check bad dimension
        # (:) used to be accepted as assumed shape -- now rank(1).
        library = ast.LibraryNode()
        node = self.library.add_function(
            "void func1(const int *array  +dimension(:))"
        )
        config = Config()
        vfy = generate.VerifyAttrs(library, config)

        vfy.check_fcn_attrs(node)
        self.assertTrue("Unable to parse dimension" in str(error.cursor.message))

    def test_implied_attrs(self):
        func = self.func1
        decls = self.func1.ast.declarator.params
        generate.check_implied_attrs(func, decls)

    def test_implied(self):
        func = self.func1
        decls = self.func1.ast.declarator.params
        expr = generate.check_implied(func, "user(array)", decls)
        self.assertEqual("user(array)", expr)

    def test_errors(self):
        func = self.func1
        decls = self.func1.ast.declarator.params

        generate.check_implied(func, "size(array2,1)", decls)

        generate.check_implied(func, "size(unknown)", decls)
        self.assertTrue("Unknown argument" in str(error.cursor.message))

        generate.check_implied(func, "len(scalar,1)", decls)
        self.assertTrue("Too many arguments" in str(error.cursor.message))


if __name__ == "__main__":
    unittest.main()
