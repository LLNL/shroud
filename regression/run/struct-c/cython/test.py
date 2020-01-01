# Copyright (c) 2017-2020, Lawrence Livermore National Security, LLC and
# other Shroud Project Developers.
# See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
# #######################################################################
#
# test the tutorial module
#
from __future__ import print_function

#import numpy as np
import unittest
import cstruct

class Cstruct(unittest.TestCase):
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

    def test_acceptStructIn(self):
        arg = cstruct.Cstruct1(1, 2.5)
        rv = cstruct.acceptStructIn(arg)
        self.assertEqual(3.5, rv)

# creating a new test suite
newSuite = unittest.TestSuite()
 
# adding a test case
newSuite.addTest(unittest.makeSuite(Cstruct))

if __name__ == "__main__":
    unittest.main()
