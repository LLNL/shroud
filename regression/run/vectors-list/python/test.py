# Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and
# other Shroud Project Developers.
# See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
# #######################################################################
#
# Test Python API generated from vectors.yaml.
# vectors-list
#
from __future__ import print_function

import numpy as np
import unittest
import vectors

class NotTrue:
    """Test bool arguments errors"""
    def __bool__(self):
        raise NotImplementedError
 
class Vectors(unittest.TestCase):
    """Test vectors problem"""
     
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

    def test_vector_sum(self):
        irv = vectors.vector_sum([1,2,3,4,5])
        self.assertEqual(15, irv)

        arg = np.array([10,20,30,40,50], dtype=np.intc)
        irv = vectors.vector_sum(arg)
        self.assertEqual(150, irv)

    def test_vector_iota_out(self):
        # The intent(out) argument is returned from the function.
        arg = vectors.vector_iota_out()
        self.assertIsInstance(arg, list)
        self.assertTrue(len(arg), 5)
        self.assertIsInstance(arg[0], int)
        self.assertEqual([1,2,3,4,5], arg)

#    ! inta is intent(out), so it will be deallocated upon entry to vector_iota_out_alloc
#    call vector_iota_out_alloc(inta)
#    call assert_true(allocated(inta))
#    call assert_equals(5 , size(inta))
#    call assert_true( all(inta == [1,2,3,4,5]), &
#         "vector_iota_out_alloc value")
#
#    ! inta is intent(inout), so it will NOT be deallocated upon entry to vector_iota_inout_alloc
#    ! Use previous value to append
#    call vector_iota_inout_alloc(inta)
#    call assert_true(allocated(inta))
#    call assert_equals(10 , size(inta))
#    call assert_true( all(inta == [1,2,3,4,5,11,12,13,14,15]), &
#         "vector_iota_inout_alloc value")
#    deallocate(inta)
#
#    intv = [1,2,3,4,5]
#    call vector_increment(intv)
#    call assert_true(all(intv(:) .eq. [2,3,4,5,6]))

    def test_vector_iota_out_d(self):
        # The intent(out) argument is returned from the function.
        arg = vectors.vector_iota_out_d()
        self.assertIsInstance(arg, list)
        self.assertTrue(len(arg), 5)
        self.assertIsInstance(arg[0], float)
        self.assertTrue(np.allclose(arg, [1,2,3,4,5]))

    def test_returnVectorAlloc(self):
        rv = vectors.ReturnVectorAlloc(10)

        self.assertIsInstance(rv, list)
        self.assertEqual(10, len(rv))
        self.assertEqual([1,2,3,4,5,6,7,8,9,10], rv)
#        self.assertTrue(np.allclose(rv, outarray))


# creating a new test suite
newSuite = unittest.TestSuite()
 
# adding a test case
newSuite.addTest(unittest.makeSuite(Vectors))

if __name__ == "__main__":
    unittest.main()
