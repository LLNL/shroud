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
import cenum

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
        self.assertEqual(10, cenum.RED)
        self.assertEqual(11, cenum.BLUE)
        self.assertEqual(12, cenum.WHITE)

    def test_enum_val(self):
        self.assertEqual(0, cenum.a1)
        self.assertEqual(3, cenum.b1)
        self.assertEqual(4, cenum.c1)
        self.assertEqual(3, cenum.d1)
        self.assertEqual(3, cenum.e1)
        self.assertEqual(4, cenum.f1)
        self.assertEqual(5, cenum.g1)
        self.assertEqual(100, cenum.h1)


# creating a new test suite
newSuite = unittest.TestSuite()
 
# adding a test case
newSuite.addTest(unittest.makeSuite(Enum))

if __name__ == "__main__":
    unittest.main()
