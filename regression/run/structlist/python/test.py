# Copyright (c) 2017-2020, Lawrence Livermore National Security, LLC and
# other Shroud Project Developers.
# See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
# #######################################################################
#
# Test Python API generated from structlist.yaml.
#
from __future__ import print_function

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

    def test_Arrays1(self):
        # derived from struct-class-c.
        # getter and setter
        # native creates Python lists.

        # name - None makes a blank string.
        # count - broadcast initial scalar.
        s = cstruct.Arrays1(name=None, count=0)
        count = s.count
        self.assertIsInstance(count, list)
        self.assertEqual(10, len(count))
        self.assertEqual(count, [0,0,0,0,0,0,0,0,0,0])

        self.assertEqual('', s.name)

        #####
        # Numpy array to constructor.
#        ref = np.array([10,20,30,40,50,60,70,80,90,100], dtype=np.intc)
#        s = cstruct.Arrays1(count=ref)
#        self.assertTrue(all(np.equal(s.count, ref)))
        
        #####
        s = cstruct.Arrays1()

        s.name = "dog"
        name = s.name
        self.assertEqual('dog', name)

        # Assign a list, NumPy will convert to array.
        ref = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        s.count = ref
        count = s.count
        self.assertIsInstance(count, list)
        self.assertEqual(10, len(count))
        self.assertEqual(count, ref)

        # getting again returns same Object.
        # XXX - will not be the same object.
#        self.assertIs(count, s.count)

        # Assign NumPy array.
        ref = [10,20,30,40,50,60,70,80,90,100]
        s.count = ref
        count = s.count
        self.assertIsInstance(count, list)
        self.assertEqual(10, len(count))
        self.assertEqual(count, ref)

        # XXX - Test assigning too few items.

        # No-op
        s.count = []

        with self.assertRaises(TypeError) as context:
            s.count = None
        self.assertTrue("argument 'count' must be iterable"
                        in str(context.exception))

        with self.assertRaises(TypeError) as context:
            s.name = 1
        self.assertTrue("argument should be string"
                        in str(context.exception))

        with self.assertRaises(TypeError) as context:
            s.count = [0, 3., "four"]
        self.assertTrue("argument 'count', index 2 must be int"
                        in str(context.exception))

        with self.assertRaises(TypeError) as context:
            s = cstruct.Arrays1(count="one")
        self.assertTrue("argument 'count', index 0 must be int"
                        in str(context.exception))

        with self.assertRaises(TypeError) as context:
            s = cstruct.Arrays1(name=10)
        self.assertTrue("argument should be string or None, not int"
                        in str(context.exception))


# creating a new test suite
newSuite = unittest.TestSuite()
 
# adding a test case
newSuite.addTest(unittest.makeSuite(Struct))

if __name__ == "__main__":
    unittest.main()
