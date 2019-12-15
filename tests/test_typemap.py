# Copyright (c) 2017-2019, Lawrence Livermore National Security, LLC and
# other Shroud Project Developers.
# See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

from __future__ import print_function

from shroud import typemap

import unittest

class Typemap(unittest.TestCase):
    def test_lookup_stmts1(self):
        statements = dict(
            result=dict(check="result"),
            result_allocatable=dict(check="result_allocatable"),
        )
        
        rv = typemap.lookup_stmts(statements, ["result"])
        self.assertEqual(rv["check"], "result")

        rv = typemap.lookup_stmts(statements, ["result", "allocatable"])
        self.assertEqual(rv["check"], "result_allocatable")

        rv = typemap.lookup_stmts(statements, ["result", ""])
        self.assertEqual(rv["check"], "result")

        rv = typemap.lookup_stmts(statements, ["result", None, "allocatable"])
        self.assertEqual(rv["check"], "result_allocatable")

        # Not found, return empty dictionary
        rv = typemap.lookup_stmts(statements, ["none"])
        self.assertEqual(rv, {})

    def test_alias(self):
        cf_tree = {}
        stmts = dict(
            a=dict(
                name="a"
            ),
            b=dict(
                name="b",
                alias="a",
            ),
        )
        typemap.update_stmt_tree(stmts, cf_tree)

        rv = typemap.lookup_stmts_tree(cf_tree, ["b"])
        self.assertIs(rv, stmts["a"])
        
    def test_lookup_tree1(self):
        cf_tree = {}
        stmts = dict(
            c_string_result_buf_allocatable=dict(
                name="c_string_result_buf_allocatable"
            ),
            c_string_scalar_result_buf_allocatable=dict(
                name="c_string_scalar_result_buf_allocatable"
            ),
        )
        typemap.update_stmt_tree(stmts, cf_tree)

        rv = typemap.lookup_stmts_tree(
            cf_tree, ["c","string","result","buf","allocatable"])
        self.assertIsNot(rv, typemap.empty_stmts)
        self.assertEqual(rv["key"], "c_string_result_buf_allocatable")

        rv = typemap.lookup_stmts_tree(
            cf_tree, ["c","string","scalar", "result","buf","allocatable"])
        self.assertIsNot(rv, typemap.empty_stmts)
        self.assertEqual(rv["key"], "c_string_scalar_result_buf_allocatable")

        # pointer is not in the tree, so skip while doing the lookup.
        rv = typemap.lookup_stmts_tree(
            cf_tree, ["c","string","pointer", "result","buf","allocatable"])
        self.assertIsNot(rv, typemap.empty_stmts)
        self.assertEqual(rv["key"], "c_string_result_buf_allocatable")
        

if __name__ == "__main__":
    unittest.main()
