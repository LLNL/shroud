# Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and
# other Shroud Project Developers.
# See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
#
# #######################################################################
#
# Test Python API generated from ccomplex.yaml.
#
from __future__ import print_function

import math
import unittest
import ccomplex


class NotTrue:
    """Test bool arguments errors"""
    def __bool__(self):
        raise NotImplementedError
 
class CComplex(unittest.TestCase):
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
     
    def test_acceptDoubleComplexInoutPtr(self):
        rv = ccomplex.acceptDoubleComplexInoutPtr(complex(1.0, 2.0))
        self.assertIsInstance(rv, complex)
        self.assertEqual(complex(3., 4.), rv)

    def test_acceptDoubleComplexOutPtr(self):
        rv = ccomplex.acceptDoubleComplexOutPtr()
        self.assertIsInstance(rv, complex)
        self.assertEqual(complex(3., 4.), rv)

    def test_acceptDoubleComplexInoutPtrFlag(self):
        rv, flag = ccomplex.acceptDoubleComplexInoutPtrFlag(complex(1.0, 2.0))
        self.assertIsInstance(rv, complex)
        self.assertEqual(complex(3., 4.), rv)
        self.assertEqual(0, flag)

    def test_acceptDoubleComplexOutPtrFlag(self):
        rv, flag = ccomplex.acceptDoubleComplexOutPtrFlag()
        self.assertIsInstance(rv, complex)
        self.assertEqual(complex(3., 4.), rv)
        self.assertEqual(0, flag)

# creating a new test suite
newSuite = unittest.TestSuite()
 
# adding a test case
newSuite.addTest(unittest.makeSuite(CComplex))

if __name__ == "__main__":
    unittest.main()
