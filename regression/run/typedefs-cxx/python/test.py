# Copyright Shroud Project Developers. See LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
#
# #######################################################################
#
# Test Python API generated from typedefs.yaml.
#
from __future__ import print_function

import unittest
import typedefs

 
class Typedefs(unittest.TestCase):
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

    def test_alias(self):
        arg1 = 10
        rv = typedefs.typefunc(arg1)
        self.assertEqual(rv, arg1 + 1)


unittest.TestLoader().loadTestsFromTestCase(Typedefs)

if __name__ == "__main__":
    unittest.main()
