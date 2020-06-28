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
import arrayclass

class Arrayclass(unittest.TestCase):
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
        arrinst = arrayclass.ArrayWrapper()
        arrinst.setSize(10)
        self.assertEqual(10, arrinst.getSize())

        isize = arrinst.fillSize()
        self.assertEqual(10, isize)

        arrinst.allocate()
        arr = arrinst.getArray()
        self.assertIsInstance(arr, np.ndarray)
        self.assertEqual('float64', arr.dtype.name)
        self.assertEqual(1, arr.ndim)
        self.assertEqual((10,), arr.shape)
        self.assertEqual(10, arr.size)

        # Make sure we're pointing to the array in the instance.
        arr[:] = 0.0
        self.assertEqual(0.0, arrinst.sumArray())
        arr[:] = 1.0
        self.assertEqual(10.0, arrinst.sumArray())
        arr[:] = 0.0
        arr[0] = 10.0
        arr[9] = 1.0
        self.assertEqual(11.0, arrinst.sumArray())

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

        arr5 = arrinst.fetchArrayPtr()
        self.assertIsInstance(arr4, np.ndarray)
        self.assertEqual('float64', arr5.dtype.name)
        self.assertEqual(1, arr5.ndim)
        self.assertEqual((10,), arr5.shape)
        self.assertEqual(10, arr5.size)

        arr6 = arrinst.fetchArrayRef()
        self.assertIsInstance(arr4, np.ndarray)
        self.assertEqual('float64', arr6.dtype.name)
        self.assertEqual(1, arr6.ndim)
        self.assertEqual((10,), arr6.shape)
        self.assertEqual(10, arr6.size)

        arr7 = arrinst.fetchArrayPtrConst()
        self.assertIsInstance(arr4, np.ndarray)
        self.assertEqual('float64', arr7.dtype.name)
        self.assertEqual(1, arr7.ndim)
        self.assertEqual((10,), arr7.shape)
        self.assertEqual(10, arr7.size)

        arr8 = arrinst.fetchArrayRefConst()
        self.assertIsInstance(arr4, np.ndarray)
        self.assertEqual('float64', arr8.dtype.name)
        self.assertEqual(1, arr8.ndim)
        self.assertEqual((10,), arr8.shape)
        self.assertEqual(10, arr8.size)

        with self.assertRaises(ValueError) as context:
            arrinst.checkPtr(None)
        self.assertTrue("called with invalid PyCapsule object"
                        in str(context.exception))

        voidptr = arrinst.fetchVoidPtr()
        self.assertEqual('PyCapsule', voidptr.__class__.__name__)
        self.assertTrue(arrinst.checkPtr(voidptr))

        voidptr = arrinst.fetchVoidRef()
        self.assertEqual('PyCapsule', voidptr.__class__.__name__)
        self.assertTrue(arrinst.checkPtr(voidptr))


# creating a new test suite
newSuite = unittest.TestSuite()
 
# adding a test case
newSuite.addTest(unittest.makeSuite(Arrayclass))

if __name__ == "__main__":
    unittest.main()
