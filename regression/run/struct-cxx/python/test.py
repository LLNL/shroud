# Copyright (c) 2017-2019, Lawrence Livermore National Security, LLC and
# other Shroud Project Developers.
# See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
#
# #######################################################################
#
# test struct.yaml
#
from __future__ import print_function

import numpy as np
import unittest
import cstruct

class Struct(unittest.TestCase):
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

    def Xtest_dtype(self):
        str = cstruct.Cstruct1()
        str.ifield = 1
        str.dfield = 2.5
        self.assertEqual(1, str.ifield)
        self.assertEqual(2.5, str.dfield)

    def Xtest_passStructByValue(self):
        i = cstruct.passStructByValue((2, 2.0))
        self.assertEqual(4, i)

        i = cstruct.passStructByValue((2.0, 2.0))
        self.assertEqual(4, i)

        with self.assertRaises(ValueError) as context:
            i = cstruct.passStructByValue((2.0, "two"))
        self.assertTrue("arg must be a 1-D array of Cstruct1" in str(context.exception))

    def Xtest_passStruct1(self):
        i = cstruct.passStruct1((12,12.6))
        self.assertEqual(12, i)

        dt = cstruct.Cstruct1_dtype
        a = np.array((1, 1.5), dtype=dt)
        i = cstruct.passStruct1(a)
        self.assertEqual(1, i)

    def Xtest_passStruct2(self):
        i, name = cstruct.passStruct2((22,22.8))
        self.assertEqual(22, i)
        self.assertEqual("passStruct2", name)

        dt = cstruct.Cstruct1_dtype
        a = np.array((1, 1.5), dtype=dt)
        i, name = cstruct.passStruct2(a)
        self.assertEqual(1, i)
        self.assertEqual("passStruct2", name)

    def Xtest_acceptStructInPtr(self):
        pass

    def Xtest_acceptStructOutPtr(self):
        str = cstruct.acceptStructOutPtr(4, 4.5)
        self.assertTrue(isinstance(str, np.ndarray))
        self.assertIs(str.dtype, cstruct.Cstruct1_dtype)
        self.assertEqual(4,   str["ifield"])
        self.assertEqual(4.5, str["dfield"])

    def Xtest_acceptStructInOutPtr(self):
        out = cstruct.acceptStructInOutPtr((22,22.8))
        self.assertTrue(isinstance(out, np.ndarray))
        self.assertIs(out.dtype, cstruct.Cstruct1_dtype)
        self.assertEqual(23,   out["ifield"])
        self.assertEqual(23.8, out["dfield"])

        dt = cstruct.Cstruct1_dtype
        a = np.array((4, 4.0), dtype=dt)
        out = cstruct.acceptStructInOutPtr(a)
        self.assertTrue(isinstance(out, np.ndarray))
        self.assertIs(out.dtype, cstruct.Cstruct1_dtype)
        self.assertEqual(5,   out["ifield"])
        self.assertEqual(5.0, out["dfield"])

    def test_returnStructByValue(self):
        out = cstruct.returnStructByValue(1, 2.5)
        self.assertTrue(isinstance(out, cstruct.Cstruct1))
        self.assertEqual(1,   out.ifield)
        self.assertEqual(2.5, out.dfield)

    def test_returnStructPtr1(self):
        out = cstruct.returnStructPtr1(33, 33.5)
        self.assertTrue(isinstance(out, cstruct.Cstruct1))
        self.assertEqual(33,   out.ifield)
        self.assertEqual(33.5, out.dfield)

    def test_returnStructPtr2(self):
        out, name = cstruct.returnStructPtr2(35, 35.5)
        self.assertTrue(isinstance(out, cstruct.Cstruct1))
        self.assertEqual(35,   out.ifield)
        self.assertEqual(35.5, out.dfield)
        self.assertEqual("returnStructPtr2", name)


# creating a new test suite
newSuite = unittest.TestSuite()
 
# adding a test case
newSuite.addTest(unittest.makeSuite(Struct))

if __name__ == "__main__":
    unittest.main()
