# Copyright (c) 2017-2020, Lawrence Livermore National Security, LLC and
# other Shroud Project Developers.
# See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
# #######################################################################
#
# test the pointers module
#
from __future__ import print_function

import math
import numpy as np
import unittest
import pointers


class NotTrue:
    """Test bool arguments errors"""
    def __bool__(self):
        raise NotImplementedError
 
class Pointers(unittest.TestCase):
    """Test pointers.yaml"""
     
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
     
    def test_intargs(self):
        self.assertEqual((1, 2), pointers.intargs(1, 2))

    def test_cos_doubles(self):
        # x = np.arange(0, 2 * np.pi, 0.1)
        inarray = [ 0.0, 0.5*np.pi, np.pi, 1.5*np.pi, 2.0*np.pi ]
        outarray = [ math.cos(v) for v in inarray]
        rv = pointers.cos_doubles(inarray)
        self.assertTrue(isinstance(rv, np.ndarray))
        self.assertEqual('float64', rv.dtype.name)
        self.assertTrue(np.allclose(rv, outarray))

    def test_truncate_to_int(self):
        rv = pointers.truncate_to_int([1.2, 2.3, 3.4, 4.5])
        self.assertTrue(isinstance(rv, np.ndarray))
        self.assertEqual('int32', rv.dtype.name)
        self.assertTrue(np.equal(rv, [1, 2, 3, 4]).all())

    def test_get_values(self):
        # out - created NumPy array.
        nout, out = pointers.get_values()
        self.assertEqual(3, nout)
        self.assertTrue(isinstance(out, np.ndarray))
        self.assertEqual('int32', out.dtype.name)
        self.assertTrue(np.equal(out, [1,2,3]).all())

    def test_get_values2(self):
        # out - created NumPy array.
        arg1, arg2 = pointers.get_values2()

        self.assertTrue(isinstance(arg1, np.ndarray))
        self.assertEqual('int32', arg1.dtype.name)
        self.assertTrue(np.equal(arg1, [1,2,3]).all())

        self.assertTrue(isinstance(arg2, np.ndarray))
        self.assertEqual('int32', arg2.dtype.name)
        self.assertTrue(np.equal(arg2, [11,12,13]).all())

    def test_Sum(self):
        self.assertEqual(15, pointers.Sum([1, 2, 3, 4, 5]))

    def test_fillIntArray(self):
        out = pointers.fillIntArray()
        self.assertTrue(isinstance(out, np.ndarray))
        self.assertEqual('int32', out.dtype.name)
        self.assertEqual([1, 2, 3], list(out))

    def test_incrementIntArray(self):
        # the argument is returned as the result because intent(INOUT)
        array = np.array([2,4,6,8], dtype=np.intc)  # int32
        out = pointers.incrementIntArray(array)
        self.assertIs(array, out)
        self.assertTrue(isinstance(out, np.ndarray))
        self.assertEqual('int32', out.dtype.name)
        self.assertTrue(np.equal(out, [3,5,7,9]).all())

        # Call with incorrect argument type
        with self.assertRaises(ValueError) as context:
            array = np.array([2,4,6,8], dtype=np.float)
            out = pointers.incrementIntArray(array)
        self.assertTrue('array must be' in str(context.exception))


# creating a new test suite
newSuite = unittest.TestSuite()
 
# adding a test case
newSuite.addTest(unittest.makeSuite(Pointers))

if __name__ == "__main__":
    unittest.main()
