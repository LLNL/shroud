# Copyright (c) 2017-2020, Lawrence Livermore National Security, LLC and
# other Shroud Project Developers.
# See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
#
# #######################################################################
#
# Test Python API generated from namespace.yaml.
#
from __future__ import print_function

import ns
import types
import unittest

class Namespace(unittest.TestCase):
    """Test namespace.yaml"""
     
    def XXsetUp(self):
        """ Setting up for the test """
        print("FooTest:setUp_:begin")
        ## do something...
        print("FooTest:setUp_:end")
     
    def XXtearDown(self):
        """Cleaning up after the test"""
        print("FooTest:tearDown_:begin")
        ## do something...
        print("FooTest:tearDown_:end")

    def test_One(self):
        ns.One()
        self.assertEqual(ns.LastFunctionCalled(), "One")

    def test_ns_outer(self):
        self.assertIsInstance(ns.outer, types.ModuleType)

    def test_outerOne(self):
        ns.outer.One()
        self.assertEqual(ns.LastFunctionCalled(), "outer::One")


# creating a new test suite
newSuite = unittest.TestSuite()
 
# adding a test case
newSuite.addTest(unittest.makeSuite(Namespace))

if __name__ == "__main__":
    unittest.main()
