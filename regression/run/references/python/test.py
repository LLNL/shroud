# Copyright (c) 2017-2020, Lawrence Livermore National Security, LLC and
# other Shroud Project Developers.
# See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
# #######################################################################
#
# test references
#
from __future__ import print_function

import numpy as np
import unittest
import references

class References(unittest.TestCase):
    """Test struct problem"""
     
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

    def test_ArrayWrapper(self):
        arr1 = references.ArrayWrapper()
        arr1.setSize(10)
        self.assertEqual(10, arr1.getSize(10))

        arr1.allocate()
        arr = arr1%getArray()
#    call assert_true(associated(arr))
#    call assert_equals(10, size(arr))


# creating a new test suite
newSuite = unittest.TestSuite()
 
# adding a test case
newSuite.addTest(unittest.makeSuite(References))

if __name__ == "__main__":
    unittest.main()
