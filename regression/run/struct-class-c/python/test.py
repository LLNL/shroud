# Copyright (c) 2017-2020, Lawrence Livermore National Security, LLC and
# other Shroud Project Developers.
# See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
# #######################################################################
#
# test struct-cxx
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

    def test_nullconstructor(self):
        str = cstruct.Cstruct1()
        self.assertEqual(0, str.ifield)
        self.assertEqual(0, str.dfield)

        # Only set second field
        str = cstruct.Cstruct1(dfield=100)
        self.assertEqual(0, str.ifield)
        self.assertEqual(100, str.dfield)

    def test_dtype(self):
        str = cstruct.Cstruct1(1, 2.5)
        self.assertEqual(1, str.ifield)
        self.assertEqual(2.5, str.dfield)

#        with self.assertRaises(TypeError) as context:
#            str = cstruct.Cstruct1(2.0, 2.5)
#        self.assertTrue("integer argument expected" in str(context.exception))

    def test_passStructByValue(self):
        str = cstruct.Cstruct1(2, 2.0)
        i = cstruct.passStructByValue(str)
        self.assertEqual(4, i)

    def test_passStruct1(self):
        str = cstruct.Cstruct1(12, 12.6)
        i = cstruct.passStruct1(str)
        self.assertEqual(12, i)

    def test_passStruct2(self):
        i, name = cstruct.passStruct2(cstruct.Cstruct1(22, 22.8))
        self.assertEqual(22, i)
        self.assertEqual("passStruct2", name)

    def test_acceptStructInPtr(self):
        pass

    def test_acceptStructOutPtr(self):
        str = cstruct.acceptStructOutPtr(4, 4.5)
        self.assertIsInstance(str, cstruct.Cstruct1)
        self.assertEqual(4,   str.ifield)
        self.assertEqual(4.5, str.dfield)

    def test_acceptStructInOutPtr(self):
        str = cstruct.Cstruct1(22, 22.8)
        out = cstruct.acceptStructInOutPtr(str)
        self.assertIs(str, out)
        self.assertEqual(23,   out.ifield)
        self.assertEqual(23.8, out.dfield)

    def test_returnStructByValue(self):
        out = cstruct.returnStructByValue(1, 2.5)
        self.assertIsInstance(out, cstruct.Cstruct1)
        self.assertEqual(1,   out.ifield)
        self.assertEqual(2.5, out.dfield)

    def test_returnConstStructByValue(self):
        out = cstruct.returnStructByValue(1, 2.5)
        self.assertIsInstance(out, cstruct.Cstruct1)
        self.assertEqual(1,   out.ifield)
        self.assertEqual(2.5, out.dfield)

    def test_returnStructPtr1(self):
        out = cstruct.returnStructPtr1(33, 33.5)
        self.assertIsInstance(out, cstruct.Cstruct1)
        self.assertEqual(33,   out.ifield)
        self.assertEqual(33.5, out.dfield)

    def test_returnStructPtr2(self):
        out, name = cstruct.returnStructPtr2(35, 35.5)
        self.assertIsInstance(out, cstruct.Cstruct1)
        self.assertEqual(35,   out.ifield)
        self.assertEqual(35.5, out.dfield)
        self.assertEqual("returnStructPtr2", name)

    def test_cstruct_ptr_create(self):
        # struct with a char * cfield
        ptr = cstruct.Cstruct_ptr()
        self.assertEqual(None, ptr.cfield)

        ptr.cfield = "standard string"
        self.assertEqual("standard string", ptr.cfield)
        ptr.cfield = u"unicode string"
        self.assertEqual("unicode string", ptr.cfield)
        ptr.cfield = b"byte string"
        self.assertEqual("byte string", ptr.cfield)

        with self.assertRaises(ValueError) as context:
            ptr.cfield = 1
        self.assertTrue("argument must be a string" in str(context.exception))

    def test_cstruct_list(self):
        # Create struct from each argument.
        iinput = [1,2,3,4,5]
        dinput = [6.,7.,8.,9.,10.]
        sinput = ["dog", "cat", "monkey", "bird", "horse"]

        s = cstruct.Cstruct_list(nitems=5, ivalue=iinput)
        self.assertEqual(iinput, s.ivalue)

        s = cstruct.Cstruct_list(nitems=5, dvalue=dinput)
        self.assertEqual(dinput, s.dvalue)

        s = cstruct.Cstruct_list(nitems=5, svalue=sinput)
        self.assertEqual(sinput, s.svalue)
        
    def test_cstruct_list_setter(self):
        # getter and setter
        s = cstruct.Cstruct_list()
        s.nitems = 5

        input = [1,2,3,4,5]
        s.ivalue = input
        ivalue = s.ivalue
        self.assertIsInstance(ivalue, list)
        self.assertEqual(5, len(ivalue))
        self.assertEqual(input, ivalue)

        input = [6,7,8,9,10]
        s.dvalue = input
        dvalue = s.dvalue
        self.assertIsInstance(dvalue, list)
        self.assertEqual(5, len(dvalue))
        self.assertEqual(input, dvalue)

        input = ["dog", "cat", "monkey", "bird", "horse"]
        s.svalue = input
        svalue = s.svalue
        self.assertIsInstance(svalue, list)
        self.assertEqual(5, len(svalue))
        self.assertEqual(input, svalue)
        
    def test_cstruct_numpy(self):
        # getter and setter
        s = cstruct.Cstruct_numpy()
        s.nitems = 5

        input = [1,2,3,4,5]
        s.ivalue = input
        ivalue = s.ivalue
        self.assertIsInstance(ivalue, np.ndarray)
        self.assertEqual('int32', ivalue.dtype.name)
        self.assertEqual(5, ivalue.size)
        self.assertTrue(all(np.equal(ivalue, input)))
#        self.assertTrue(np.allclose(ivalue, outarray))

        input = [6,7,8,9,10]
        s.dvalue = input
        dvalue = s.dvalue
        self.assertIsInstance(dvalue, np.ndarray)
        self.assertEqual('float64', dvalue.dtype.name)
        self.assertEqual(5, dvalue.size)
        self.assertTrue(all(np.equal(dvalue, input)))

        # Get back the same NumPy array assigned
        input = np.array([10,20,30,40,50], dtype=np.intc)
        s.ivalue = input
        ivalue = s.ivalue
        self.assertIs(ivalue, input)


# creating a new test suite
newSuite = unittest.TestSuite()
 
# adding a test case
newSuite.addTest(unittest.makeSuite(Struct))

if __name__ == "__main__":
    unittest.main()
