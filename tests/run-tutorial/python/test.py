# Copyright (c) 2017-2018, Lawrence Livermore National Security, LLC. 
# Produced at the Lawrence Livermore National Laboratory 
#
# LLNL-CODE-738041.
# All rights reserved. 
#
# This file is part of Shroud.  For details, see
# https://github.com/LLNL/shroud. Please also read shroud/LICENSE.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#
# * Redistributions of source code must retain the above copyright
#   notice, this list of conditions and the disclaimer below.
# 
# * Redistributions in binary form must reproduce the above copyright
#   notice, this list of conditions and the disclaimer (as noted below)
#   in the documentation and/or other materials provided with the
#   distribution.
#
# * Neither the name of the LLNS/LLNL nor the names of its contributors
#   may be used to endorse or promote products derived from this
#   software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL LAWRENCE
# LIVERMORE NATIONAL SECURITY, LLC, THE U.S. DEPARTMENT OF ENERGY OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# #######################################################################
#
# test the tutorial module
#
from __future__ import print_function

import numpy as np
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

    def test_enum_Direction(self):
        # enum values
        self.assertEqual(2, tutorial.Class1.UP)
        self.assertEqual(3, tutorial.Class1.DOWN)
        self.assertEqual(100, tutorial.Class1.LEFT)
        self.assertEqual(101, tutorial.Class1.RIGHT)

        obj = tutorial.Class1()
        # class method with enums
        self.assertEqual(tutorial.Class1.LEFT, obj.directionFunc(tutorial.Class1.LEFT))

        # module method with enums
        self.assertEqual(tutorial.Class1.RIGHT, tutorial.directionFunc(tutorial.Class1.LEFT))
     
    # test routine A
    def testFunction1(self):
        tutorial.Function1()

    def testFunction2(self):
        rv_double = tutorial.Function2(1.0, 4)
        self.assertEqual(rv_double, 5.0)#, "A is not equal to B")

    def testFunction3(self):
        rv_logical = tutorial.Function3(False)
        self.assertTrue(rv_logical)

        # Should any object which resolved to True or False be accepted?
        # if 0:    is legal
        self.assertRaises(TypeError, tutorial.Function3, 0)

    def testReturnIntPtr(self):
        rv = tutorial.ReturnIntPtr()
        self.assertTrue(isinstance(rv, np.ndarray))
        self.assertEqual('int32', rv.dtype.name)
        self.assertTrue(all(np.equal(rv, [1,2,3,4,5,6,7])))

    def testFunction4a(self):
        rv_char = tutorial.Function4a("dog", "cat")
        self.assertEqual(rv_char, "dogcat")

        # result as argument not needed for Python
#    call function4b("dog", "cat", rv_char)
#    call assert_true( rv_char == "dogcat")

    def testFunction5(self):
        rv_double = tutorial.Function5()
        self.assertAlmostEqual(rv_double, 13.1415)
        rv_double = tutorial.Function5(1.0)
        self.assertAlmostEqual(rv_double, 11.0)

        rv_double = tutorial.Function5(1.0, False)
        self.assertAlmostEqual(rv_double, 1.0)

    def testFunction6(self):
        tutorial.Function6("name")
        self.assertEqual(tutorial.LastFunctionCalled(), "Function6(string)")

        tutorial.Function6(1)
        self.assertEqual(tutorial.LastFunctionCalled(), "Function6(int)")

        self.assertRaises(TypeError, tutorial.Function6, 1.0)

    def XXX_test_Function7_8(self):
        tutorial.Function7(1)
        self.assertEqual(tutorial.LastFunctionCalled(), "Function7<int>")
        tutorial.Function7(10.0)
        self.assertEqual(tutorial.LastFunctionCalled(), "Function7<double>")

        # return values set by calls to function7
        rv = tutorial.Function8_int()
        self.assertEqual(rv, 1)
        rv = tutorial.Function8_double()
        self.assertEqual(rv, 10.0)

    def test_Function9(self):
        # This has fortran_generic attribute but you get that for free in Python
        tutorial.Function9(1)
        tutorial.Function9(1.0)

    def test_Function10(self):
        # overloaded (no default args)
        tutorial.Function10()
        tutorial.Function10("foo", 1.0)
        tutorial.Function10("bar", 1.0)

    def testsum(self):
        self.assertEqual(15, tutorial.Sum([1, 2, 3, 4, 5]))

    def test_overload1(self):
        self.assertEqual(10, tutorial.overload1(10))
        self.assertEqual(10, tutorial.overload1(1., 10))

        self.assertEqual(142, tutorial.overload1(10,11,12))
        self.assertEqual(142, tutorial.overload1(1., 10,11,12))

        self.assertRaises(TypeError, tutorial.overload1, 1.0)
        self.assertRaises(TypeError, tutorial.overload1, "dog")
        
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

    def test_class1_create1(self):
        obj = tutorial.Class1()
        self.assertTrue(isinstance(obj, tutorial.Class1))
        self.assertEqual(0, obj.m_flag)
        del obj

    def test_class1_create2(self):
        obj = tutorial.Class1(1)
        self.assertTrue(isinstance(obj, tutorial.Class1))
        self.assertEqual(1, obj.m_flag)
        del obj

    def test_class1_method1(self):
        obj0 = tutorial.Class1()
        self.assertEqual(0, obj0.Method1())

        obj1 = tutorial.Class1(1)
        self.assertEqual(1, obj1.Method1())

    def test_class1_equivalent(self):
        obj0 = tutorial.Class1()
        obj1 = tutorial.Class1(1)
        self.assertTrue(obj0.equivalent(obj0))
        self.assertFalse(obj0.equivalent(obj1))

    def test_class1_useclass(self):
        obj0 = tutorial.Class1()
        self.assertEqual(0, tutorial.useclass(obj0))

        # getclass2 is const, not wrapped yet

        obj0a = tutorial.getclass3()
        self.assertTrue(isinstance(obj0a, tutorial.Class1))

    def test_class1_useclass_error(self):
        """Pass illegal argument to useclass"""
        obj0 = tutorial.Class1()
        self.assertRaises(TypeError, tutorial.useclass(obj0))

    def test_singleton(self):
        # it'd be cool if obj0 is obj1
        obj0 = tutorial.Singleton.getReference()
        obj1 = tutorial.Singleton.getReference()

        obj2 = obj0.getReference()

# creating a new test suite
newSuite = unittest.TestSuite()
 
# adding a test case
newSuite.addTest(unittest.makeSuite(Tutorial))

if __name__ == "__main__":
    unittest.main()
