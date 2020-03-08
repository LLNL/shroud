# Copyright (c) 2017-2020, Lawrence Livermore National Security, LLC and
# other Shroud Project Developers.
# See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
# #######################################################################
#
# test struct.yaml
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

    def test_class(self):
        "struct as class"
        a = cstruct.Cstruct_as_class(1,2)
        self.assertEqual(1,   a.x1)
        self.assertEqual(2,   a.y1)

    def test_dtype(self):
        """struct as numpy
        """
        dt = cstruct.Cstruct_as_numpy_dtype
        #print("Byte order is:",dt.byteorder) 
        #print("Size is:",dt.itemsize) 
        self.assertEqual(dt.names, ('x2', 'y2'))
        self.assertEqual(dt.char, 'V')
        self.assertEqual("void64", dt.name) 
        self.assertEqual("int32", dt["x2"].name)
        self.assertEqual("int32", dt["y2"].name)
        a = np.array([(1, 2), (3, 4)], dtype=dt) 
        self.assertEqual(1,   a.ndim)
        self.assertEqual(2,   a.size)
        self.assertEqual(1,   a[0]["x2"])
        self.assertEqual(2,   a[0]["y2"])
        self.assertEqual(3,   a[1]["x2"])
        self.assertEqual(4,   a[1]["y2"])


# creating a new test suite
newSuite = unittest.TestSuite()
 
# adding a test case
newSuite.addTest(unittest.makeSuite(Struct))

if __name__ == "__main__":
    unittest.main()
