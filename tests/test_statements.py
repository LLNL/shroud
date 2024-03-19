# Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and
# other Shroud Project Developers.
# See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

from __future__ import print_function

from shroud import statements
from shroud import util

import unittest

default_stmts = statements.default_stmts

class Statements(unittest.TestCase):
    def XXXtest_alias(self):
        # Prefix names with "c" to work with statements.default_stmts.
        cf_dict = {}
        stmts = [
            dict(
                name="c_a",
            ),
            dict(
                name="c_b",
                alias="c_a",
            ),
        ]
        statements.update_stmt_tree(
            stmts, cf_dict, cf_tree, statements.default_stmts)

        rv = statements.lookup_stmts_tree(cf_tree, ["c", "b"])
        self.assertIsInstance(rv, util.Scope)
        self.assertEqual("c_a", rv.name)
        
    def test_base(self):
        # Prefix names with "c" to work with statements.default_stmts.
        cf_dict = {}
        stmts = [
            dict(
                name="c_mixin_a",
                field1="field1_from_c_a",
                field2="field2_from_c_a",
            ),
            dict(
                name="c_a",
                mixin=[
                    "c_mixin_a",
                ],
            ),
            dict(
                name="c_b",
                mixin=[
                    "c_mixin_a",
                ],
                field2="field2_from_c_b",
            ),
        ]
        statements.process_mixin(stmts, default_stmts, cf_dict)

        rv = cf_dict.get("c_a")
        self.assertIsInstance(rv, util.Scope)
        self.assertEqual("field1_from_c_a", rv.field1)
        self.assertEqual("field2_from_c_a", rv.field2)

        rv = cf_dict.get("c_b")
        self.assertIsInstance(rv, util.Scope)
        self.assertEqual("field1_from_c_a", rv.field1)
        self.assertEqual("field2_from_c_b", rv.field2)
        
    def test_mixin(self):
        # Prefix names with "c" to work with statements.default_stmts.
        cf_dict = {}
        stmts = [
            dict(
                name="c_mixin_field1",
                field1="field1_from_mixin_field1",
                field1a="field1a_from_mixin_field1",
            ),
            dict(
                name="c_mixin_field2",
                field2="field2_from_mixin_field2",
                field2a="field2a_from_mixin_field2",
            ),
            dict(
                name="c_a",
                field1="field1_from_c_a",
                field2="field2_from_c_a",
            ),
            dict(
                name="c_b",
                mixin=["c_mixin_field1", "c_mixin_field2"],
                field2="field2_from_c_b",
            ),
        ]
        statements.process_mixin(stmts, default_stmts, cf_dict)

        rv = cf_dict.get("c_a")
        self.assertIsInstance(rv, util.Scope)
        self.assertEqual("field1_from_c_a", rv.field1)
        self.assertEqual("field2_from_c_a", rv.field2)

        rv = cf_dict.get("c_b")
        self.assertIsInstance(rv, util.Scope)
        self.assertEqual("field1_from_mixin_field1", rv.field1)
        self.assertEqual("field1a_from_mixin_field1", rv.field1a)
        self.assertEqual("field2_from_c_b", rv.field2)
        
    def test_lookup_tree1(self):
        cf_dict = {}
        stmts = [
            dict(
                name="c_string_result_buf_allocatable"
            ),
            dict(
                name="c_string_scalar_result_buf_allocatable"
            ),
        ]
        statements.process_mixin(stmts, default_stmts, cf_dict)

        rv = cf_dict.get("c_string_result_buf_allocatable")
        self.assertEqual(rv["name"], "c_string_result_buf_allocatable")

        rv = cf_dict.get("c_string_scalar_result_buf_allocatable")
        self.assertEqual(rv["name"], "c_string_scalar_result_buf_allocatable")

        # pointer is not in the tree
        self.assertIsNone(cf_dict.get("c_string_pointer_result_buf_allocatable"))
        

if __name__ == "__main__":
    unittest.main()
