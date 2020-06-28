# Copyright (c) 2017-2020, Lawrence Livermore National Security, LLC and
# other Shroud Project Developers.
# See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

from __future__ import print_function

from shroud import typemap
from shroud import util

import unittest

class Typemap(unittest.TestCase):
    def XXXtest_alias(self):
        # Prefix names with "c" to work with typemap.default_stmts.
        cf_tree = {}
        stmts = [
            dict(
                name="c_a",
            ),
            dict(
                name="c_b",
                alias="c_a",
            ),
        ]
        typemap.update_stmt_tree(stmts, cf_tree, typemap.default_stmts)

        rv = typemap.lookup_stmts_tree(cf_tree, ["c", "b"])
        self.assertIsInstance(rv, util.Scope)
        self.assertEqual("c_a", rv.name)
        
    def test_base(self):
        # Prefix names with "c" to work with typemap.default_stmts.
        cf_tree = {}
        stmts = [
            dict(
                name="c_a",
                field1="field1_from_c_a",
                field2="field2_from_c_a",
            ),
            dict(
                name="c_b",
                base="c_a",
                field2="field2_from_c_b",
            ),
        ]
        typemap.update_stmt_tree(stmts, cf_tree, typemap.default_stmts)

        rv = typemap.lookup_stmts_tree(cf_tree, ["c", "a"])
        self.assertIsInstance(rv, util.Scope)
        self.assertEqual("field1_from_c_a", rv.field1)
        self.assertEqual("field2_from_c_a", rv.field2)

        rv = typemap.lookup_stmts_tree(cf_tree, ["c", "b"])
        self.assertIsInstance(rv, util.Scope)
        self.assertEqual("field1_from_c_a", rv.field1)
        self.assertEqual("field2_from_c_b", rv.field2)
        
    def test_lookup_tree1(self):
        cf_tree = {}
        stmts = [
            dict(
                name="c_string_result_buf_allocatable"
            ),
            dict(
                name="c_string_scalar_result_buf_allocatable"
            ),
        ]
        typemap.update_stmt_tree(stmts, cf_tree, typemap.default_stmts)

        rv = typemap.lookup_stmts_tree(
            cf_tree, ["c","string","result","buf","allocatable"])
        self.assertEqual(rv["name"], "c_string_result_buf_allocatable")

        rv = typemap.lookup_stmts_tree(
            cf_tree, ["c","string","scalar", "result","buf","allocatable"])
        self.assertEqual(rv["name"], "c_string_scalar_result_buf_allocatable")

        # pointer is not in the tree, so skip while doing the lookup.
        rv = typemap.lookup_stmts_tree(
            cf_tree, ["c","string","pointer", "result","buf","allocatable"])
        self.assertEqual(rv["name"], "c_string_result_buf_allocatable")
        

if __name__ == "__main__":
    unittest.main()
