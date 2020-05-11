# Copyright (c) 2017-2020, Lawrence Livermore National Security, LLC and
# other Shroud Project Developers.
# See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
# #######################################################################
#
# Test Python API generated from references.yaml.
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
        arrinst = references.ArrayWrapper()
        arrinst.setSize(10)
        self.assertEqual(10, arrinst.getSize())

        arrinst.allocate()
        arr = arrinst.getArray()
        self.assertIsInstance(arr, np.ndarray)
        self.assertEqual('float64', arr.dtype.name)
        self.assertEqual(1, arr.ndim)
        self.assertEqual((10,), arr.shape)
        self.assertEqual(10, arr.size)

        arrconst = arrinst.getArrayConst()
        self.assertIsInstance(arrconst, np.ndarray)
        self.assertEqual('float64', arrconst.dtype.name)
        self.assertEqual(1, arrconst.ndim)
        self.assertEqual((10,), arrconst.shape)
        self.assertEqual(10, arrconst.size)

        # Both getArray and getArrayConst return a NumPy array to the
        # same pointer. But a new array is created each time.
        self.assertIsNot(arr, arrconst)

        arr3 = arrinst.getArrayC()
        self.assertIsInstance(arr3, np.ndarray)
        self.assertEqual('float64', arr3.dtype.name)
        self.assertEqual(1, arr3.ndim)
        self.assertEqual((10,), arr3.shape)
        self.assertEqual(10, arr3.size)

        arr4 = arrinst.getArrayConstC()
        self.assertIsInstance(arr4, np.ndarray)
        self.assertEqual('float64', arr4.dtype.name)
        self.assertEqual(1, arr4.ndim)
        self.assertEqual((10,), arr4.shape)
        self.assertEqual(10, arr4.size)


# creating a new test suite
newSuite = unittest.TestSuite()
 
# adding a test case
newSuite.addTest(unittest.makeSuite(References))

if __name__ == "__main__":
    unittest.main()
