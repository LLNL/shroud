# Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and
# other Shroud Project Developers.
# See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
########################################################################

from __future__ import print_function

from shroud import ast
from shroud import error
from shroud import fcfmt
from shroud import util

import unittest

error.get_cursor()

class WFormat(unittest.TestCase):
    def test_arg_cxx_int(self):
        library = ast.LibraryNode()
        func = library.add_function("void func1(int *array)")

        arg = func.ast.declarator.params[0]

        fmt_var = util.Scope(
            None,
            typemap=arg.typemap,
            c_var="arg1",
            cxx_var="cxx_var_name",
            cxx_other="other_name",
        )
        fmtarg = fcfmt.FormatGen(func, arg, fmt_var)
        self.assertEqual("array", str(fmtarg))
        self.assertEqual("array", fmtarg.name)
        self.assertEqual("cxx", fmtarg.language)
#        print(4, fmtarg.nonconst_addr)
#        print(5, fmtarg.nonconst_addr.cxx_var)
#        print(6, fmtarg.nonconst_addr.cxx_other)
#        print(6, fmtarg.nonconst_addr.cxx_local)

#        print(7, fmtarg.no_such_attr)

#        print(8, fmtarg.tester)
#        print(10, fmtarg.__name)

        self.assertEqual("int *", str(fmtarg.cxxdecl))
        self.assertEqual("int *arg1", fmtarg.cxxdecl.c_var)
        self.assertEqual("int *===>xxx<===", fmtarg.cxxdecl.xxx)

        # cidecl
#        print(11, fmtarg.cxxdecl)         # abstract
#        print(11, fmtarg.cxxdecl.cxx_var)

    def test_arg_cxx_const_int(self):
        library = ast.LibraryNode()
        func = library.add_function("void func1(const int *array)")

        arg = func.ast.declarator.params[0]

        fmt_var = util.Scope(
            None,
            typemap=arg.typemap,
            c_var="arg1",
            cxx_var="cxx_var_name",
            cxx_other="other_name",
        )
        fmtarg = fcfmt.FormatGen(func, arg, fmt_var)
        self.assertEqual("array", str(fmtarg))
        self.assertEqual("array", fmtarg.name)
        self.assertEqual("const int *", str(fmtarg.cxxdecl))
        self.assertEqual("const int *arg1", fmtarg.cxxdecl.c_var)
        self.assertEqual("const int *===>xxx<===", fmtarg.cxxdecl.xxx)

    def test_arg_cxx_vector(self):
        library = ast.LibraryNode()
        func = library.add_function("void func1(vector<int> *array)")

        arg = func.ast.declarator.params[0]

        fmt_var = util.Scope(
            None,
            typemap=arg.typemap,
            c_var="arg1",
            cxx_var="cxx_var_name",
            cxx_other="other_name",
        )
        fmtarg = fcfmt.FormatGen(func, arg, fmt_var)
        self.assertEqual("array", str(fmtarg))
        self.assertEqual("array", fmtarg.name)
        self.assertEqual("vector<int> *", str(fmtarg.cxxdecl))
        self.assertEqual("vector<int> *arg1", fmtarg.cxxdecl.c_var)
        self.assertEqual("vector<int> *===>xxx<===", fmtarg.cxxdecl.xxx)
    
if __name__ == "__main__":
    unittest.main()
        
