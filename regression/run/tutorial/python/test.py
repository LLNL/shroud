# Copyright (c) 2017-2020, Lawrence Livermore National Security, LLC and
# other Shroud Project Developers.
# See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
#
# #######################################################################
#
# Test Python API generated from tutorial.yaml.
#
from __future__ import print_function

import unittest
import tutorial

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

    def test_enum_Color(self):
        self.assertEqual(0, tutorial.RED)
        self.assertEqual(1, tutorial.BLUE)
        self.assertEqual(2, tutorial.WHITE)

        # pass and return enumeration
        self.assertEqual(tutorial.RED, tutorial.colorfunc(tutorial.BLUE))

    def test_NoReturnNoArguments(self):
        tutorial.NoReturnNoArguments()

    def test_PassByValue(self):
        rv_double = tutorial.PassByValue(1.0, 4)
        self.assertEqual(rv_double, 5.0)#, "A is not equal to B")

    def test_ConcatenateStrings(self):
        rv_char = tutorial.ConcatenateStrings("dog", "cat")
        self.assertEqual(rv_char, "dogcat")

    def test_UseDefaultArguments(self):
        rv_double = tutorial.UseDefaultArguments()
        self.assertAlmostEqual(rv_double, 13.1415)
        rv_double = tutorial.UseDefaultArguments(1.0)
        self.assertAlmostEqual(rv_double, 11.0)

        rv_double = tutorial.UseDefaultArguments(1.0, False)
        self.assertAlmostEqual(rv_double, 1.0)

    def test_OverloadedFunction(self):
        tutorial.OverloadedFunction("name")
        self.assertEqual(tutorial.LastFunctionCalled(),
                         "OverloadedFunction(string)")

        tutorial.OverloadedFunction(1)
        self.assertEqual(tutorial.LastFunctionCalled(),
                         "OverloadedFunction(int)")

        self.assertRaises(TypeError, tutorial.OverloadedFunction, 1.0)

    def test_TemplateArgument_8(self):
        """Test cxx_template"""
        tutorial.TemplateArgument(1)
        self.assertEqual(tutorial.LastFunctionCalled(), "TemplateArgument<int>")
        tutorial.TemplateArgument(10.0)
        self.assertEqual(tutorial.LastFunctionCalled(), "TemplateArgument<double>")

        # return values set by calls to TemplateArgument
        #rv = tutorial.TemplateReturn_int()
        #self.assertEqual(rv, 1)
        #rv = tutorial.TemplateReturn_double()
        #self.assertEqual(rv, 10.0)

    def test_FortranGenericOverloaded(self):
        # overloaded (no default args)
        tutorial.FortranGenericOverloaded()
        tutorial.FortranGenericOverloaded("foo", 1.0)
        tutorial.FortranGenericOverloaded("bar", 1.0)

    def test_UseDefaultOverload(self):
        self.assertEqual(10, tutorial.UseDefaultOverload(10))
        self.assertEqual(10, tutorial.UseDefaultOverload(1., 10))

        self.assertEqual(142, tutorial.UseDefaultOverload(10,11,12))
        self.assertEqual(142, tutorial.UseDefaultOverload(1., 10,11,12))

        self.assertRaises(TypeError, tutorial.UseDefaultOverload, 1.0)
        self.assertRaises(TypeError, tutorial.UseDefaultOverload, "dog")
        
    def test_typefunc(self):
        self.assertEqual(2, tutorial.typefunc(2))

    def test_enumfunc(self):
        self.assertEqual(2, tutorial.enumfunc(1))

    def test_getMinMax(self):
        r = tutorial.getMinMax()
        self.assertEqual((-1,100), r)

#
#  end subroutine test_functions
#

# creating a new test suite
newSuite = unittest.TestSuite()
 
# adding a test case
newSuite.addTest(unittest.makeSuite(Tutorial))

if __name__ == "__main__":
    unittest.main()
