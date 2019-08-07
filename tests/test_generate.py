
from __future__ import print_function

from shroud import ast
from shroud import generate

import unittest


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

    def test_implied_attrs(self):
        decls = self.func1.ast.params
        generate.check_implied_attrs(decls)

    def test_implied(self):
        decls = self.func1.ast.params
        expr = generate.check_implied("user(array)", decls)
        self.assertEqual("user(array)", expr)

    def test_errors(self):
        decls = self.func1.ast.params

        with self.assertRaises(RuntimeError) as context:
            generate.check_implied("size(array,n2)", decls)
        self.assertTrue("Too many arguments" in str(context.exception))

        with self.assertRaises(RuntimeError) as context:
            generate.check_implied("size(array2)", decls)
        self.assertTrue("Unknown argument" in str(context.exception))

        with self.assertRaises(RuntimeError) as context:
            generate.check_implied("size(scalar)", decls)
        self.assertTrue(
            "must have dimension attribute" in str(context.exception)
        )

        with self.assertRaises(RuntimeError) as context:
            generate.check_implied("len(scalar,1)", decls)
        self.assertTrue("Too many arguments" in str(context.exception))


if __name__ == "__main__":
    unittest.main()
