# Copyright (c) 2017-2020, Lawrence Livermore National Security, LLC and
# other Shroud Project Developers.
# See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

from __future__ import print_function

from shroud import ast
from shroud import util
from shroud import wrapp

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

        node._fmtargs = dict(
            array=dict(fmtpy=util.Scope(None, py_var="SHPy_array")),
            scalar=dict(fmtpy=util.Scope(None, py_var="SHPy_scalar")),
        )
        self.func1 = node

    def test_implied1(self):
        self.assertEqual(
            "PyArray_SIZE(SHPy_array)",
            wrapp.py_implied("size(array)", self.func1),
        )
        self.assertEqual(
            "PyArray_SIZE(SHPy_array)+2",
            wrapp.py_implied("size(array) + 2", self.func1),
        )

    def test_expr1(self):
        self.assertEqual("size+n", wrapp.py_implied("size+n", self.func1))


class CheckStruct(unittest.TestCase):
    def setUp(self):
        self.library = ast.LibraryNode()
        self.struct = self.library.add_struct("""
struct Cstruct_list {
    int nitems;
    int *ivalue     +dimension(nitems+nitems);
    double *dvalue  +dimension(nitems*TWO);
    char **svalue   +dimension(nitems);
};
""")

    def test_dimension(self):
        self.struct.create_node_map()
        map = self.struct.map_name_to_node

        var = map['ivalue']
        var.fmtdict.PY_struct_context = "struct."
        dims = wrapp.py_struct_dimension(self.struct, var)
        self.assertEqual("struct.nitems+struct.nitems", dims)

        var = map['dvalue']
        var.fmtdict.PY_struct_context = "struct."
        dims = wrapp.py_struct_dimension(self.struct, var)
        self.assertEqual("struct.nitems*TWO", dims)


if __name__ == "__main__":
    unittest.main()
