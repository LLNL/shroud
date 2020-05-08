# Copyright (c) 2017-2020, Lawrence Livermore National Security, LLC and
# other Shroud Project Developers.
# See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
#
# #######################################################################
#
# Test Python API generated from enum.yaml.
#
from __future__ import print_function

import unittest
import enum

class NotTrue:
    """Test bool arguments errors"""
    def __bool__(self):
        raise NotImplementedError
 
class Enum(unittest.TestCase):
    """Test tutorial problem"""
     
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

    def test_enum_Color(self):
        self.assertEqual(10, enum.RED)
        self.assertEqual(11, enum.BLUE)
        self.assertEqual(12, enum.WHITE)

    def test_enum_val(self):
        self.assertEqual(0, enum.a1)
        self.assertEqual(3, enum.b1)
        self.assertEqual(4, enum.c1)
        self.assertEqual(3, enum.d1)
        self.assertEqual(3, enum.e1)
        self.assertEqual(4, enum.f1)
        self.assertEqual(5, enum.g1)
        self.assertEqual(100, enum.h1)


# creating a new test suite
newSuite = unittest.TestSuite()
 
# adding a test case
newSuite.addTest(unittest.makeSuite(Enum))

if __name__ == "__main__":
    unittest.main()
