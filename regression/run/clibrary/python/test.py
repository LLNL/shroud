# Copyright (c) 2018-2019, Lawrence Livermore National Security, LLC. 
#
# Produced at the Lawrence Livermore National Laboratory 
#
# LLNL-CODE-738041.
#
# All rights reserved. 
#
# This file is part of Shroud.
#
# For details about use and distribution, please read LICENSE.
#
# #######################################################################
#
# test the clibrary module
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
     
    def testPassByValue(self):
        self.assertEqual(5.0, clibrary.PassByValue(1.0, 4))

    def testPassByReference(self):
        rv = clibrary.PassByReference(3.14)
        self.assertEqual(3, rv)

    def testfunction3(self):
        self.assertEqual(True, clibrary.Function3(False))

    def testcheckBool(self):
        self.assertEqual((False, False), clibrary.checkBool(True, True))

    def testfunction4a(self):
        self.assertEqual('dogcat', clibrary.Function4a('dog', 'cat'))

    def testacceptName(self):
        clibrary.acceptName('spot')
##        self.assertEqual('acceptName', clibrary.last_function_called())

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

    def testsum(self):
        self.assertEqual(15, clibrary.Sum([1, 2, 3, 4, 5]))


# creating a new test suite
newSuite = unittest.TestSuite()
 
# adding a test case
newSuite.addTest(unittest.makeSuite(Tutorial))

if __name__ == "__main__":
    unittest.main()
