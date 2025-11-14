# Copyright Shroud Project Developers. See LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

from shroud import ast
from shroud import declast
from shroud import metaattrs
from shroud import statements
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

        params = node.ast.declarator.params

        bind_arg = statements.fetch_arg_bind(node, params[0], "py")
        fmt_arg = statements.set_bind_fmtdict(bind_arg, node.fmtdict)
        fmt_arg.py_var = "SHPy_array"

        bind_arg = statements.fetch_arg_bind(node, params[1], "py")
        fmt_arg = statements.set_bind_fmtdict(bind_arg, node.fmtdict)
        fmt_arg.py_var = "SHPy_scalar"
        
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
        metaattrs.process_metaattrs(self.library, "share")
        metaattrs.process_metaattrs(self.library, "py")

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
        # done in metaattrs.FillMeta.parse_dim_attrs
        declarator = var.ast.declarator
        a_bind = statements.fetch_var_bind(var, "share")
        meta = a_bind.meta
        meta["dim_ast"] = declast.check_dimension(declarator.attrs["dimension"])
        fmt = var.fmtdict
        fmt.PY_struct_context = "struct."
        have_array = wrapp.py_struct_dimension(self.struct, var, fmt)
        self.assertEqual("1", fmt.rank)
        self.assertEqual("struct.nitems+struct.nitems", fmt.npy_intp_values)
        self.assertEqual("struct.nitems+struct.nitems", fmt.npy_intp_size)

        #####
        var = map['dvalue']
        # done in metaattrs.FillMeta.parse_dim_attrs
        declarator = var.ast.declarator
        a_bind = statements.fetch_var_bind(var, "share")
        meta = a_bind.meta
        meta["dim_ast"] = declast.check_dimension(declarator.attrs["dimension"])
        
        fmt = var.fmtdict
        fmt.PY_struct_context = "struct."
        have_array = wrapp.py_struct_dimension(self.struct, var, fmt)
        self.assertEqual("1", fmt.rank)
        self.assertEqual("struct.nitems*TWO", fmt.npy_intp_values)
        self.assertEqual("struct.nitems*TWO", fmt.npy_intp_size)


if __name__ == "__main__":
    unittest.main()
