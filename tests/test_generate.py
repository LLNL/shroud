# Copyright (c) 2017-2020, Lawrence Livermore National Security, LLC and
# other Shroud Project Developers.
# See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

from __future__ import print_function

from shroud import ast
from shroud import generate

import unittest

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
            "int  scalar +intent(in)+implied(size(array))"
            ")"
        )
        self.func1 = node

    def test_dimension_1(self):
        # Check missing dimension value
        # (:) used to be accepted as assumed shape -- now rank(1).
        library = ast.LibraryNode()
        node = self.library.add_function(
            "void func1(const int *array  +dimension)"
        )
        config = Config()
        vfy = generate.VerifyAttrs(library, config)

        with self.assertRaises(RuntimeError) as context:
            vfy.check_fcn_attrs(node)
        self.assertTrue("dimension attribute must have a value" in str(context.exception))

    def test_dimension_2(self):
        # Check bad dimension
        # (:) used to be accepted as assumed shape -- now rank(1).
        library = ast.LibraryNode()
        node = self.library.add_function(
            "void func1(const int *array  +dimension(:))"
        )
        config = Config()
        vfy = generate.VerifyAttrs(library, config)

        with self.assertRaises(RuntimeError) as context:
            vfy.check_fcn_attrs(node)
        self.assertTrue("Unable to parse dimension" in str(context.exception))

    def test_implied_attrs(self):
        func = self.func1
        decls = self.func1.ast.params
        generate.check_implied_attrs(func, decls)

    def test_implied(self):
        func = self.func1
        decls = self.func1.ast.params
        expr = generate.check_implied(func, "user(array)", decls)
        self.assertEqual("user(array)", expr)

    def test_errors(self):
        func = self.func1
        decls = self.func1.ast.params

        with self.assertRaises(RuntimeError) as context:
            generate.check_implied(func, "size(array,n2)", decls)
        self.assertTrue("Too many arguments" in str(context.exception))

        with self.assertRaises(RuntimeError) as context:
            generate.check_implied(func, "size(array2)", decls)
        self.assertTrue("Unknown argument" in str(context.exception))

        with self.assertRaises(RuntimeError) as context:
            generate.check_implied(func, "len(scalar,1)", decls)
        self.assertTrue("Too many arguments" in str(context.exception))


if __name__ == "__main__":
    unittest.main()
