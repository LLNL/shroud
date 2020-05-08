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
import classes

class NotTrue:
    """Test bool arguments errors"""
    def __bool__(self):
        raise NotImplementedError
 
class Classes(unittest.TestCase):
    """Test classes problem"""
     
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

    def test_enum_Direction(self):
        # enum values
        self.assertEqual(2, classes.Class1.UP)
        self.assertEqual(3, classes.Class1.DOWN)
        self.assertEqual(100, classes.Class1.LEFT)
        self.assertEqual(101, classes.Class1.RIGHT)

        obj = classes.Class1()
        # class method with enums
        self.assertEqual(classes.Class1.LEFT, obj.directionFunc(classes.Class1.LEFT))

        # module method with enums
        self.assertEqual(classes.Class1.RIGHT, classes.directionFunc(classes.Class1.LEFT))
     
    def test_class1_create1(self):
        obj = classes.Class1()
        self.assertIsInstance(obj, classes.Class1)
        self.assertEqual(0, obj.test)
        obj.test = 4
        self.assertEqual(4, obj.test)
        # test -1 since PyInt_AsLong returns -1 on error
        obj.test = -1
        self.assertEqual(-1, obj.test)
        with self.assertRaises(AttributeError) as context:
            obj.m_flag = 1
        self.assertTrue("is not writable" in str(context.exception))
        with self.assertRaises(TypeError) as context:
            obj.test = "dog"
        self.assertTrue("an integer is required" in str(context.exception))
        del obj

    def test_class1_create2(self):
        obj = classes.Class1(1)
        self.assertIsInstance(obj, classes.Class1)
        self.assertEqual(1, obj.m_flag)
        del obj

    def test_class1_method1(self):
        obj0 = classes.Class1()
        self.assertEqual(0, obj0.Method1())

        obj1 = classes.Class1(1)
        self.assertEqual(1, obj1.Method1())

    def test_class1_equivalent(self):
        obj0 = classes.Class1()
        obj1 = classes.Class1(1)
        self.assertTrue(obj0.equivalent(obj0))
        self.assertFalse(obj0.equivalent(obj1))

    def test_class1_PassClassByValue(self):
        # passClassByValue sets the global retrived by get_global_flag()
        classes.set_global_flag(0)
        obj0 = classes.Class1()
        obj0.test = 13
        classes.passClassByValue(obj0)
        self.assertEqual(13, classes.get_global_flag())

    def test_class1_useclass(self):
        obj0 = classes.Class1()
        self.assertEqual(0, classes.useclass(obj0))

        # getclass2 is const, not wrapped yet

        obj0a = classes.getclass3()
        self.assertIsInstance(obj0a, classes.Class1)

    def test_class1_useclass_error(self):
        """Pass illegal argument to useclass"""
        obj0 = classes.Class1()
        self.assertRaises(TypeError, classes.useclass(obj0))

    def test_singleton(self):
        # XXX - it'd be cool if obj0 is obj1
        obj0 = classes.Singleton.getReference()
        obj1 = classes.Singleton.getReference()

        obj2 = obj0.getReference()

# creating a new test suite
newSuite = unittest.TestSuite()
 
# adding a test case
newSuite.addTest(unittest.makeSuite(Classes))

if __name__ == "__main__":
    unittest.main()
