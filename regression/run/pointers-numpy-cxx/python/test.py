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
        pointers.intargs_in(5)            # set global_int.
        iargout = pointers.intargs_out()  # get global_int
        self.assertEqual(5, iargout)
    
        iarginout = pointers.intargs_inout(6)  # set global_int
        self.assertEqual(6, pointers.intargs_out())
        self.assertEqual(7, iarginout)
        
        self.assertEqual((1, 2), pointers.intargs(1, 2))

    def test_cos_doubles(self):
        # x = np.arange(0, 2 * np.pi, 0.1)
        inarray = [ 0.0, 0.5*np.pi, np.pi, 1.5*np.pi, 2.0*np.pi ]
        outarray = [ math.cos(v) for v in inarray]
        rv = pointers.cos_doubles(inarray)
        self.assertIsInstance(rv, np.ndarray)
        self.assertEqual('float64', rv.dtype.name)
        self.assertTrue(np.allclose(rv, outarray))

    def test_truncate_to_int(self):
        rv = pointers.truncate_to_int([1.2, 2.3, 3.4, 4.5])
        self.assertIsInstance(rv, np.ndarray)
        self.assertEqual('int32', rv.dtype.name)
        self.assertTrue(np.equal(rv, [1, 2, 3, 4]).all())

    def test_get_values(self):
        # out - created NumPy array.
        nout, out = pointers.get_values()
        self.assertEqual(3, nout)
        self.assertIsInstance(out, np.ndarray)
        self.assertEqual('int32', out.dtype.name)
        self.assertTrue(np.equal(out, [1,2,3]).all())

    def test_get_values2(self):
        # out - created NumPy array.
        arg1, arg2 = pointers.get_values2()

        self.assertIsInstance(arg1, np.ndarray)
        self.assertEqual('int32', arg1.dtype.name)
        self.assertTrue(np.equal(arg1, [1,2,3]).all())

        self.assertIsInstance(arg2, np.ndarray)
        self.assertEqual('int32', arg2.dtype.name)
        self.assertTrue(np.equal(arg2, [11,12,13]).all())

    def test_iota_allocatable(self):
        # out - created list.
        out = pointers.iota_allocatable(3)
        self.assertIsInstance(out, np.ndarray)
        self.assertEqual('int32', out.dtype.name)
        self.assertEqual(3, len(out))
        self.assertTrue(np.equal(out, [1,2,3]).all())

    def XXXtest_iota_dimension(self):
        # out - created list.
        out = pointers.iota_dimension(3)
        self.assertIsInstance(out, list)
        self.assertEqual(3, len(out))
        self.assertEqual([1,2,3], out)

    def test_Sum(self):
        self.assertEqual(15, pointers.Sum([1, 2, 3, 4, 5]))

    def test_fillIntArray(self):
        out = pointers.fillIntArray()
        self.assertIsInstance(out, np.ndarray)
        self.assertEqual('int32', out.dtype.name)
        self.assertEqual([1, 2, 3], list(out))

    def test_incrementIntArray(self):
        # the argument is returned as the result because intent(INOUT)
        array = np.array([2,4,6,8], dtype=np.intc)  # int32
        out = pointers.incrementIntArray(array)
        self.assertIs(array, out)
        self.assertIsInstance(out, np.ndarray)
        self.assertEqual('int32', out.dtype.name)
        self.assertTrue(np.equal(out, [3,5,7,9]).all())

        # Call with incorrect argument type
        with self.assertRaises(ValueError) as context:
            array = np.array([2,4,6,8], dtype=np.float)
            out = pointers.incrementIntArray(array)
        self.assertTrue('array must be' in str(context.exception))

    def test_fill_with_zeros(self):
        # swig test
        array = np.array([2,4,6,8], dtype=np.double)
        pointers.fill_with_zeros(array)
        self.assertTrue(np.equal(array, 0.0).all())

    def test_accumulate(self):
        # swig test
        array = np.array([1,2,3,4,5], dtype=np.intc)  # int32
        sum = pointers.accumulate(array)
        self.assertEqual(15, sum)
        
    def test_acceptCharArrayIn(self):
        n = pointers.acceptCharArrayIn(["dog", "cat", "monkey"])
        self.assertEqual(len("dog"), n)

    def test_out_ptrs(self):
        # Functions which return a pointer in an argument.

        pointers.setGlobalInt(0)
#        ptr1 = pointers.getPtrToScalar()
#        #        call assert_equals(0, iscalar)
#
#        # iscalar points to global_int in pointers.c.
        pointers.setGlobalInt(5)
#        #call assert_equals(5, iscalar)
#
        p = None
        p = pointers.getPtrToFixedArray()
        self.assertIsInstance(p, np.ndarray)
        self.assertEqual('int32', p.dtype.name)
        self.assertEqual(10, p.size)
        self.assertEqual(0, pointers.sumFixedArray())
        # Make sure we're assigning to global_array.
        p[0] = 1
        p[9] = 2
        self.assertEqual(3, pointers.sumFixedArray())

        # Returns global_array in pointers.c.
        p = None
        p = pointers.getPtrToDynamicArray()
        self.assertIsInstance(p, np.ndarray)
        self.assertEqual('int32', p.dtype.name)
        self.assertEqual(10, p.size)
        
        # Returns global_array in pointers.c.
        p = None
        p = pointers.getPtrToFuncArray()
        self.assertIsInstance(p, np.ndarray)
        self.assertEqual('int32', p.dtype.name)
        self.assertEqual(10, p.size)

        
#        p = None
#        p = pointers.getPtrToConstScalar()
#        self.assertIsInstance(p, np.ndarray)
#        self.assertEqual('int32', p.dtype.name)
#        self.assertEqual(10, p.size)

        p = None
        p = pointers.getPtrToFixedConstArray()
        self.assertIsInstance(p, np.ndarray)
        self.assertEqual('int32', p.dtype.name)
        self.assertEqual(10, p.size)

        p = None
        p = pointers.getPtrToDynamicConstArray()
        self.assertIsInstance(p, np.ndarray)
        self.assertEqual('int32', p.dtype.name)
        self.assertEqual(10, p.size)

        p = None
        p = pointers.getRawPtrToScalar()
        self.assertEqual('PyCapsule', p.__class__.__name__)
#        call assert_true(c_associated(cptr_scalar))
#        # associated with global_int in pointers.c
#        call assert_true(c_associated(cptr_scalar, c_loc(iscalar)))
#
        p = None
        p = pointers.getRawPtrToFixedArray()
        self.assertEqual('PyCapsule', p.__class__.__name__)
#        call assert_true(c_associated(cptr_array))
#        # associated with global_fixed_array in pointers.c
#        call assert_true(c_associated(cptr_array, c_loc(iarray)))

    def test_void_ptr_func(self):
        void = None
        void = pointers.returnAddress1(1)
        self.assertEqual('PyCapsule', void.__class__.__name__)

        void = None
        void = pointers.returnAddress2(1)
        self.assertEqual('PyCapsule', void.__class__.__name__)

        void = None
        void = pointers.fetchVoidPtr()
        self.assertEqual('PyCapsule', void.__class__.__name__)

    def test_return_ptr(self):
        ptr = pointers.returnIntPtrToScalar()
        self.assertIsInstance(ptr, np.ndarray)
        self.assertEqual('int32', ptr.dtype.name)
        self.assertEqual(1, ptr.size)
        
        ptr = pointers.returnIntPtrToFixedArray()
        self.assertIsInstance(ptr, np.ndarray)
        self.assertEqual('int32', ptr.dtype.name)
        self.assertEqual(10, ptr.size)

        ptr = pointers.returnIntPtrToConstScalar()
        self.assertIsInstance(ptr, np.ndarray)
        self.assertEqual('int32', ptr.dtype.name)
        self.assertEqual(1, ptr.size)

        ptr = pointers.returnIntPtrToFixedConstArray()
        self.assertIsInstance(ptr, np.ndarray)
        self.assertEqual('int32', ptr.dtype.name)
        self.assertEqual(10, ptr.size)

        # +deref(scalar), NumPy array is not returned.
        pointers.setGlobalInt(8)
        val = pointers.returnIntScalar()
        self.assertIsInstance(val, int)
        self.assertEqual(8, val)


# creating a new test suite
newSuite = unittest.TestSuite()
 
# adding a test case
newSuite.addTest(unittest.makeSuite(Pointers))

if __name__ == "__main__":
    unittest.main()
