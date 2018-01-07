from __future__ import print_function

from shroud import ast

import unittest

class CheckAst(unittest.TestCase):
#    maxDiff = None

    def test_a_library1(self):
        """Test LibraryNode"""
        library = ast.LibraryNode()

        self.assertEqual(library.language, 'c++')
        self.assertEqual(library.options.wrap_c, True)
        self.assertEqual(library.options.wrap_fortran, True)

        fmt = library._fmt
        self.assertEqual(fmt.C_prefix, 'DEF_')


    def test_a_library1(self):
        """Update LibraryNode"""
        node = dict(
            language='c',
            options=dict(
                wrap_c=False,
                C_prefix='XXX_',
            )
        )
        library = ast.LibraryNode(node)

        self.assertEqual(library.language, 'c')              # updated from dict
        self.assertEqual(library.options.wrap_c, False)      # updated from dict
        self.assertEqual(library.options.wrap_fortran, True)

        fmt = library._fmt
        self.assertEqual(fmt.C_prefix, 'XXX_')

