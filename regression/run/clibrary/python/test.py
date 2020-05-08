# Copyright (c) 2017-2020, Lawrence Livermore National Security, LLC and
# other Shroud Project Developers.
# See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
#
# #######################################################################
#
# Test Python API generated from clibrary.yaml.
#
from __future__ import print_function

import math
import unittest
import clibrary


class NotTrue:
    """Test bool arguments errors"""
    def __bool__(self):
        raise NotImplementedError
 
class Tutorial(unittest.TestCase):
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
     
    def testNoReturnNoArguments(self):
        clibrary.NoReturnNoArguments()

    def testPassByValue(self):
        self.assertEqual(5.0, clibrary.PassByValue(1.0, 4))

    def testPassByReference(self):
        rv = clibrary.PassByReference(3.14)
        self.assertEqual(3, rv)

    def testcheckBool(self):
        self.assertEqual((False, False), clibrary.checkBool(True, True))

    def testfunction4a(self):
        self.assertEqual('dogcat', clibrary.Function4a('dog', 'cat'))

    def testacceptName(self):
        clibrary.acceptName('spot')
##        self.assertEqual('acceptName', clibrary.last_function_called())

    def testpassCharPtrInOut(self):
        """char * +intent(out)"""
        self.assertEqual('DOG', clibrary.passCharPtrInOut('dog'))

    def testReturnOneName(self):
        name1 = clibrary.returnOneName()
        self.assertEqual("bill", name1)

    def testReturnTwoNames(self):
        name1, name2 = clibrary.returnTwoNames()
        self.assertEqual("tom", name1)
        self.assertEqual("frank", name2)

    def testImpliedTextLen(self):
        text = clibrary.ImpliedTextLen()
        self.assertEqual("ImpliedTextLen", text)

    def testImpliedLen(self):
        rv_int = clibrary.ImpliedLen("bird  ")
        self.assertEqual(6, rv_int)

    def testImpliedLen2(self):
        # XXX - this should return 4
        rv_int = clibrary.ImpliedLenTrim("bird  ")
        self.assertEqual(6, rv_int)

    def testImpliedBoolTrue(self):
        self.assertTrue(clibrary.ImpliedBoolTrue())

    def testImpliedBoolFalse(self):
        self.assertFalse(clibrary.ImpliedBoolFalse())


# creating a new test suite
newSuite = unittest.TestSuite()
 
# adding a test case
newSuite.addTest(unittest.makeSuite(Tutorial))

if __name__ == "__main__":
    unittest.main()
