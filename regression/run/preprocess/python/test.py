# Copyright (c) 2017-2020, Lawrence Livermore National Security, LLC and
# other Shroud Project Developers.
# See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
# #######################################################################
#
# Test Python API generated from preprocess.yaml.
#
from __future__ import print_function

import numpy as np
import unittest
import preprocess

class NotTrue:
    """Test bool arguments errors"""
    def __bool__(self):
        raise NotImplementedError
 
class Preprocess(unittest.TestCase):
    """Test preprocess problem"""
     
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

    def test_method1(self):
        obj = preprocess.User1()
        obj.method1()

    def test_method2(self):
        """Method User1.method2 has been conditionally compiled out."""
        obj = preprocess.User1()
        with self.assertRaises(AttributeError):
            obj.method2()

    def test_User2(self):
        """Class User2 has been conditionally compiled out."""
        obj = preprocess.User1()
        with self.assertRaises(AttributeError):
            preprocess.User2()


# creating a new test suite
newSuite = unittest.TestSuite()
 
# adding a test case
newSuite.addTest(unittest.makeSuite(Preprocess))

if __name__ == "__main__":
    unittest.main()
