# Copyright (c) 2017-2020, Lawrence Livermore National Security, LLC and
# other Shroud Project Developers.
# See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
# #######################################################################
#
# Test Python API generated from pointers.yaml.
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
        self.assertIsInstance(rv, list)
        self.assertTrue(np.allclose(rv, outarray))

    def test_truncate_to_int(self):
#        argin = [1.2, 2.3, 3.4, 4.5]
#        rv = pointers.truncate_to_int(argin)
        rv = pointers.truncate_to_int([1.2, 2.3, 3.4, 4.5])
        self.assertIsInstance(rv, list)
        self.assertEqual([1, 2, 3, 4], rv)

    def test_get_values(self):
        # out - created list.
        nout, out = pointers.get_values()
        self.assertEqual(3, nout)
        self.assertIsInstance(out, list)
        self.assertEqual([1,2,3], out)

    def test_get_values2(self):
        # out - created list.
        arg1, arg2 = pointers.get_values2()

        self.assertIsInstance(arg1, list)
        self.assertEqual([1,2,3], arg1)

        self.assertIsInstance(arg2, list)
        self.assertEqual([11,12,13], arg2)

    def test_iota_allocatable(self):
        # out - created list.
        out = pointers.iota_allocatable(3)
        self.assertIsInstance(out, list)
        self.assertEqual(3, len(out))
        self.assertEqual([1,2,3], out)

    def test_iota_dimension(self):
        # out - created list.
        out = pointers.iota_dimension(3)
        self.assertIsInstance(out, list)
        self.assertEqual(3, len(out))
        self.assertEqual([1,2,3], out)

    def test_Sum(self):
        self.assertEqual(15, pointers.Sum([1, 2, 3, 4, 5]))

    def test_fillIntArray(self):
        out = pointers.fillIntArray()
        self.assertIsInstance(out, list)
        self.assertEqual([1, 2, 3], out)

    def test_incrementIntArray(self):
        # the argument is returned as the result because intent(INOUT)
        array = [2,4,6,8]
        out = pointers.incrementIntArray(array)
#        self.assertIs(array, out)
        self.assertIsInstance(out, list)
        self.assertEqual([3,5,7,9], out)

        # Call with incorrect argument type
        with self.assertRaises(TypeError) as context:
            array = 7
            out = pointers.incrementIntArray(array)
        self.assertTrue('must be iterable' in str(context.exception))

        with self.assertRaises(TypeError) as context:
            array = [ 1, "two", 3 ]
            out = pointers.incrementIntArray(array)
        self.assertTrue('must be int' in str(context.exception))

        # test with another iteratable
        array = np.array([2,4,6,8], dtype=np.intc)  # int32
        out = pointers.incrementIntArray(array)
#        self.assertIs(array, out)
        self.assertIsInstance(out, list)
        self.assertEqual([3,5,7,9], out)

        out = pointers.incrementIntArray((2,4,6,8))
#        self.assertIs(array, out)
        self.assertIsInstance(out, list)
        self.assertEqual([3,5,7,9], out)

    def test_acceptCharArrayIn(self):
        pointers.acceptCharArrayIn(["dog", "cat", "monkey"])

    def test_void_ptr_func(self):
        void = None
        void = pointers.returnAddress1(1)
        self.assertEqual('PyCapsule', void.__class__.__name__)

        void = None
        void = pointers.returnAddress2(1)
        self.assertEqual('PyCapsule', void.__class__.__name__)

    def test_return_ptr(self):
        # Pointers return a list.
        ptr = pointers.returnIntPtrToScalar()
        self.assertIsInstance(ptr, list)
        self.assertEqual(1, len(ptr))
        
        ptr = pointers.returnIntPtrToFixedArray()
        self.assertIsInstance(ptr, list)
        self.assertEqual(10, len(ptr))

        ptr = pointers.returnIntPtrToConstScalar()
        self.assertIsInstance(ptr, list)
        self.assertEqual(1, len(ptr))

        ptr = pointers.returnIntPtrToFixedConstArray()
        self.assertIsInstance(ptr, list)
        self.assertEqual(10, len(ptr))


# creating a new test suite
newSuite = unittest.TestSuite()
 
# adding a test case
newSuite.addTest(unittest.makeSuite(Pointers))

if __name__ == "__main__":
    unittest.main()
