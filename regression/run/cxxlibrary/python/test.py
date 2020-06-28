# Copyright (c) 2017-2020, Lawrence Livermore National Security, LLC and
# other Shroud Project Developers.
# See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
# #######################################################################
#
# Test Python API generated from struct.yaml.
# struct-numpy-c
#
from __future__ import print_function

import numpy as np
import unittest
import cxxlibrary

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

    def test_StructNumpy(self):
        # NumPy creates a struct out of the tuple
        # which is returned since the argument is intent(inout)
        i, str1out = cxxlibrary.passStructByReference((2, 2.0))
        self.assertEqual(4, i)
        self.assertEqual(3, str1out["ifield"])

        # Create struct via numpy
        dt = cxxlibrary.Cstruct1_dtype
        str1 = np.array((3, 2.0), dtype=dt)
        
        rvi = cxxlibrary.passStructByReferenceIn(str1) # assign global_Cstruct1
        self.assertEqual(6, rvi)
        str2 = cxxlibrary.passStructByReferenceOut()   # fetch global_Cstruct1
        self.assertEqual(str1, str2)

        # Change str1 in place.
        str3 = cxxlibrary.passStructByReferenceInout(str1)
        self.assertEqual(4, str1["ifield"])

    def test_StructClass(self):
        str1 = cxxlibrary.Cstruct1_cls(2, 2.0)

        # Argument is intent(inout), return input struct as output.
        i, str1out = cxxlibrary.passStructByReferenceCls(str1)
        self.assertEqual(4, i)
        self.assertEqual(3, str1.ifield)
        self.assertIs(str1, str1out)

        rvi = cxxlibrary.passStructByReferenceInCls(str1) # assign global_Cstruct1
        self.assertEqual(6, rvi)
        str2 = cxxlibrary.passStructByReferenceOutCls()   # fetch global_Cstruct1
        self.assertEqual(str1.ifield, str2.ifield)
        self.assertEqual(str1.dfield, str2.dfield)

        # Change str1 in place.
        str3 = cxxlibrary.passStructByReferenceInoutCls(str1)
        self.assertEqual(4, str1.ifield)

    def test_DefaultArgs(self):
        self.assertTrue(cxxlibrary.defaultPtrIsNULL())
        self.assertFalse(cxxlibrary.defaultPtrIsNULL([1., 2.]))

# creating a new test suite
newSuite = unittest.TestSuite()
 
# adding a test case
newSuite.addTest(unittest.makeSuite(Struct))

if __name__ == "__main__":
    unittest.main()
