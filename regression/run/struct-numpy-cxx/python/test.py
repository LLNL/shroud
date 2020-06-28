# Copyright (c) 2017-2020, Lawrence Livermore National Security, LLC and
# other Shroud Project Developers.
# See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
# #######################################################################
#
# Test Python API generated from struct.yaml.
# struct-numpy-cxx
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

    def test_dtype(self):
        dt = cstruct.Cstruct1_dtype
        #print("Byte order is:",dt.byteorder) 
        #print("Size is:",dt.itemsize) 
        self.assertEqual(dt.names, ('ifield', 'dfield'))
        self.assertEqual(dt.char, 'V')
        self.assertEqual("void128", dt.name) 
        self.assertEqual("int32", dt["ifield"].name)
        self.assertEqual("float64", dt["dfield"].name)
        a = np.array([(1, 1.5), (2, 2.6)], dtype=dt) 
        self.assertEqual(1,   a.ndim)
        self.assertEqual(2,   a.size)
        self.assertEqual(1,   a[0]["ifield"])
        self.assertEqual(1.5, a[0]["dfield"])
        self.assertEqual(2,   a[1]["ifield"])
        self.assertEqual(2.6, a[1]["dfield"])

    def test_passStructByValue(self):
        i = cstruct.passStructByValue((2, 2.0))
        self.assertEqual(4, i)

        i = cstruct.passStructByValue((2.0, 2.0))
        self.assertEqual(4, i)

        with self.assertRaises(ValueError) as context:
            i = cstruct.passStructByValue((2.0, "two"))
        self.assertTrue("arg must be a 1-D array of STR_cstruct1" in str(context.exception))

    def test_passStruct1(self):
        i = cstruct.passStruct1((12,12.6))
        self.assertEqual(12, i)

        dt = cstruct.Cstruct1_dtype
        a = np.array((1, 1.5), dtype=dt)
        i = cstruct.passStruct1(a)
        self.assertEqual(1, i)

    def test_passStruct2(self):
        i, name = cstruct.passStruct2((22,22.8))
        self.assertEqual(22, i)
        self.assertEqual("passStruct2", name)

        dt = cstruct.Cstruct1_dtype
        a = np.array((1, 1.5), dtype=dt)
        i, name = cstruct.passStruct2(a)
        self.assertEqual(1, i)
        self.assertEqual("passStruct2", name)

    def test_acceptStructInPtr(self):
        s = np.array((3,3.0), dtype=cstruct.Cstruct1_dtype)
        cstruct.acceptStructInPtr(s)

    def test_acceptStructOutPtr(self):
        str = cstruct.acceptStructOutPtr(4, 4.5)
        self.assertIsInstance(str, np.ndarray)
        self.assertIs(str.dtype, cstruct.Cstruct1_dtype)
        self.assertEqual(4,   str["ifield"])
        self.assertEqual(4.5, str["dfield"])

    def test_acceptStructInOutPtr(self):
        out = cstruct.acceptStructInOutPtr((22,22.8))
        self.assertIsInstance(out, np.ndarray)
        self.assertIs(out.dtype, cstruct.Cstruct1_dtype)
        self.assertEqual(23,   out["ifield"])
        self.assertEqual(23.8, out["dfield"])

        dt = cstruct.Cstruct1_dtype
        a = np.array((4, 4.0), dtype=dt)
        out = cstruct.acceptStructInOutPtr(a)
        self.assertIsInstance(out, np.ndarray)
        self.assertIs(out.dtype, cstruct.Cstruct1_dtype)
        self.assertEqual(5,   out["ifield"])
        self.assertEqual(5.0, out["dfield"])

    def test_returnStructByValue(self):
        out = cstruct.returnStructByValue(1, 2.5)
        self.assertIsInstance(out, np.ndarray)
        self.assertIs(out.dtype, cstruct.Cstruct1_dtype)
        self.assertEqual(0, out.ndim)
        self.assertEqual(1, out.size)
        self.assertEqual(1,   out["ifield"])
        self.assertEqual(2.5, out["dfield"])

    def test_returnConstStructByValue(self):
        out = cstruct.returnConstStructByValue(1, 2.5)
        self.assertIsInstance(out, np.ndarray)
        self.assertIs(out.dtype, cstruct.Cstruct1_dtype)
        self.assertEqual(0, out.ndim)
        self.assertEqual(1, out.size)
        self.assertEqual(1,   out["ifield"])
        self.assertEqual(2.5, out["dfield"])

    def test_returnStructPtr1(self):
        out = cstruct.returnStructPtr1(33, 33.5)
        self.assertIsInstance(out, np.ndarray)
        self.assertIs(out.dtype, cstruct.Cstruct1_dtype)
        self.assertEqual(0, out.ndim)
        self.assertEqual(1, out.size)
        self.assertEqual(33,   out["ifield"])
        self.assertEqual(33.5, out["dfield"])

    def test_returnStructPtr2(self):
        out, name = cstruct.returnStructPtr2(35, 35.5)
        self.assertIsInstance(out, np.ndarray)
        self.assertIs(out.dtype, cstruct.Cstruct1_dtype)
        self.assertEqual(0, out.ndim)
        self.assertEqual(1, out.size)
        self.assertEqual(35,   out["ifield"])
        self.assertEqual(35.5, out["dfield"])
        self.assertEqual("returnStructPtr2", name)


# creating a new test suite
newSuite = unittest.TestSuite()
 
# adding a test case
newSuite.addTest(unittest.makeSuite(Struct))

if __name__ == "__main__":
    unittest.main()
