# Copyright (c) 2017-2019, Lawrence Livermore National Security, LLC and
# other Shroud Project Developers.
# See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
#
# #######################################################################
#
# test the vectors module
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

    def test_returnVectorAlloc(self):
        rv = vectors.returnVectorAlloc(10)

        self.assertIsInstance(rv, np.ndarray)
        self.assertEqual('int32', rv.dtype.name)
        self.assertEqual(10, rv.size)
        self.assertEqual(1, rv)
        self.assertTrue(all(np.equal(rv, [1,2,3,4,5,6,7,8,9,10])))
#        self.assertTrue(np.allclose(rv, outarray))


# creating a new test suite
newSuite = unittest.TestSuite()
 
# adding a test case
newSuite.addTest(unittest.makeSuite(Vectors))

if __name__ == "__main__":
    unittest.main()
