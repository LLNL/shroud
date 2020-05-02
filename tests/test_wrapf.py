# Copyright (c) 2017-2020, Lawrence Livermore National Security, LLC and
# other Shroud Project Developers.
# See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

from __future__ import print_function

from shroud import ast
from shroud import util
from shroud import wrapf

import unittest


class CheckAllocatable(unittest.TestCase):
    def setUp(self):
        # Create a dictionary of parsed arguments
        self.library = ast.LibraryNode()
        node = self.library.add_function(
            "void func1("
            "int *in1 +intent(in)+dimension(:),"
            "int *in2 +intent(in)+dimension(:,:),"
            "int *out +intent(out),"
            "int flag)"
        )

        node._fmtargs = dict(
            in1=dict(fmtf=util.Scope(None, f_var="in1")),
            in2=dict(fmtf=util.Scope(None, f_var="in2")),
            out=dict(fmtf=util.Scope(None, f_var="out")),
            flag=dict(fmtf=util.Scope(None, f_var="flag")),
        )
        self.func1 = node

    def test_errors(self):
        self.library.options.F_standard = 2003
        pre_call = []

        with self.assertRaises(RuntimeError) as context:
            wrapf.attr_allocatable(
                "mold=none", self.func1, self.func1.ast.params[2], pre_call
            )
        self.assertTrue("does not exist" in str(context.exception))

        with self.assertRaises(RuntimeError) as context:
            wrapf.attr_allocatable(
                "mold=flag", self.func1, self.func1.ast.params[2], pre_call
            )
        self.assertTrue(
            "must have dimension or rank attribute" in str(context.exception)
        )

    def test_allocatable1d(self):
        self.library.options.F_standard = 2003
        pre_call = []
        wrapf.attr_allocatable(
            "mold=in1", self.func1, self.func1.ast.params[2], pre_call
        )
        self.assertEqual(
            "allocate(out(lbound(in1,1):ubound(in1,1)))", pre_call[0]
        )

        self.library.options.F_standard = 2008
        pre_call = []
        wrapf.attr_allocatable(
            "mold=in1", self.func1, self.func1.ast.params[2], pre_call
        )
        self.assertEqual("allocate(out, mold=in1)", pre_call[0])

    def test_allocatable2d(self):
        self.library.options.F_standard = 2003
        pre_call = []
        wrapf.attr_allocatable(
            "mold=in2", self.func1, self.func1.ast.params[2], pre_call
        )
        self.assertEqual(
            "allocate(out(lbound(in2,1):ubound(in2,1),"
            "lbound(in2,2):ubound(in2,2)))",
            pre_call[0],
        )

        self.library.options.F_standard = 2008
        pre_call = []
        wrapf.attr_allocatable(
            "mold=in2", self.func1, self.func1.ast.params[2], pre_call
        )
        self.assertEqual("allocate(out, mold=in2)", pre_call[0])


if __name__ == "__main__":
    unittest.main()
