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


if __name__ == "__main__":
    unittest.main()
