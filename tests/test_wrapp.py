# Copyright (c) 2017-2020, Lawrence Livermore National Security, LLC and
# other Shroud Project Developers.
# See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

from __future__ import print_function

from shroud import ast
from shroud import declast
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

        ##### Scalar
        var = map['nitems']
        fmt = var.fmtdict
        have_array = wrapp.py_struct_dimension(self.struct, var, fmt)
        self.assertEqual("0", fmt.rank)
        self.assertEqual("1", fmt.npy_intp_values)
        self.assertEqual("1", fmt.npy_intp_size)

        #####
        var = map['ivalue']
        # done in generate.VerifyAttrs.parse_attrs
        var.ast.metaattrs["dimension"] = \
            declast.check_dimension(var.ast.attrs["dimension"])
        
        fmt = var.fmtdict
        fmt.PY_struct_context = "struct."
        have_array = wrapp.py_struct_dimension(self.struct, var, fmt)
        self.assertEqual("1", fmt.rank)
        self.assertEqual("struct.nitems+struct.nitems", fmt.npy_intp_values)
        self.assertEqual("struct.nitems+struct.nitems", fmt.npy_intp_size)

        #####
        var = map['dvalue']
        # done in generate.VerifyAttrs.parse_attrs
        var.ast.metaattrs["dimension"] = \
            declast.check_dimension(var.ast.attrs["dimension"])
        
        fmt = var.fmtdict
        fmt.PY_struct_context = "struct."
        have_array = wrapp.py_struct_dimension(self.struct, var, fmt)
        self.assertEqual("1", fmt.rank)
        self.assertEqual("struct.nitems*TWO", fmt.npy_intp_values)
        self.assertEqual("struct.nitems*TWO", fmt.npy_intp_size)


if __name__ == "__main__":
    unittest.main()
