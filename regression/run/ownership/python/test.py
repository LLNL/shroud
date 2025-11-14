# Copyright Shroud Project Developers. See LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
# #######################################################################
#
# Test Python API generated from ownership.yaml.
#

import numpy as np
import unittest
import ownership

class Ownership(unittest.TestCase):
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

    #----------------------------------------
    # return scalar

    def testReturnIntPtrScalar(self):
        "Return pointer as int python scalar"
        # deref(scalar)
        rv = ownership.ReturnIntPtrScalar()
        self.assertIsInstance(rv, int)
        self.assertEqual(10, rv)

    def testReturnIntPtrPointer(self):
        "Return pointer to int numpy scalar"
        # deref(pointer)
        rv = ownership.ReturnIntPtrPointer()
        self.assertIsInstance(rv, np.ndarray)
        self.assertEqual('int32', rv.dtype.name)
        self.assertEqual(1, rv.size)
        self.assertEqual(1, rv)

    #----------------------------------------
    # return dimension(len) owner(caller)

    def testReturnIntPtrDimDefault(self):
        "Return pointer to existing int array"
        rv = ownership.ReturnIntPtrDimDefault()
        self.assertIsInstance(rv, np.ndarray)
        self.assertEqual('int32', rv.dtype.name)
        self.assertEqual(7, rv.size)
        self.assertTrue(all(np.equal(rv, [31,32,33,34,35,36,37])))

    #----------------------------------------
    # return dimension(len) owner(library)

    def testReturnIntPtrDimDefaultNew(self):
        "Return pointer to a new int array"
        rv = ownership.ReturnIntPtrDimDefaultNew()
        self.assertIsInstance(rv, np.ndarray)
        self.assertEqual('int32', rv.dtype.name)
        self.assertEqual(5, rv.size)
        self.assertTrue(all(np.equal(rv, [30,31,32,33,34])))


unittest.TestLoader().loadTestsFromTestCase(Ownership)

if __name__ == "__main__":
    unittest.main()
