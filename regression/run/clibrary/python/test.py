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
     
    def testfunction2(self):
        self.assertEqual(5.0, clibrary.Function2(1.0, 4))

    def testfunction3(self):
        self.assertEqual(True, clibrary.Function3(False))

    def testfunction3b(self):
        self.assertEqual((False, False), clibrary.Function3b(True, True))

    def testfunction4a(self):
        self.assertEqual('dogcat', clibrary.Function4a('dog', 'cat'))

    def testsum(self):
        self.assertEqual(15, clibrary.Sum([1, 2, 3, 4, 5]))


# creating a new test suite
newSuite = unittest.TestSuite()
 
# adding a test case
newSuite.addTest(unittest.makeSuite(Tutorial))

if __name__ == "__main__":
    unittest.main()
