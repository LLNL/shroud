# Copyright (c) 2017-2019, Lawrence Livermore National Security, LLC and
# other Shroud Project Developers.
# See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

from __future__ import print_function

from shroud import typemap

import unittest

class Typemap(unittest.TestCase):
    def setUp(self):
        self.statements = dict(
            result=dict(check="result"),
            result_allocatable=dict(check="result_allocatable"),
        )
    
    def test_lookup_stmts1(self):
        rv = typemap.lookup_stmts(self.statements, ["result"])
        self.assertEqual(rv["check"], "result")

        rv = typemap.lookup_stmts(self.statements, ["result", "allocatable"])
        self.assertEqual(rv["check"], "result_allocatable")

        rv = typemap.lookup_stmts(self.statements, ["result", ""])
        self.assertEqual(rv["check"], "result")

        rv = typemap.lookup_stmts(self.statements, ["result", None, "allocatable"])
        self.assertEqual(rv["check"], "result_allocatable")

        # Not found, return empty dictionary
        rv = typemap.lookup_stmts(self.statements, ["none"])
        self.assertEqual(rv, {})


if __name__ == "__main__":
    unittest.main()
