
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

    def test_errors(self):
        with self.assertRaises(RuntimeError) as context:
            generate.check_implied("bad(array)", self.func1)
        self.assertTrue("Unexpected function" in str(context.exception))

        with self.assertRaises(RuntimeError) as context:
            generate.check_implied("size(array,n2)", self.func1)
        self.assertTrue("Too many arguments" in str(context.exception))

        with self.assertRaises(RuntimeError) as context:
            generate.check_implied("size(array2)", self.func1)
        self.assertTrue("Unknown argument" in str(context.exception))

        with self.assertRaises(RuntimeError) as context:
            generate.check_implied("size(scalar)", self.func1)
        self.assertTrue("must have dimension attribute" in str(context.exception))


if __name__ == "__main__":
    unittest.main()
