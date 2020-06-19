# Copyright (c) 2017-2020, Lawrence Livermore National Security, LLC and
# other Shroud Project Developers.
# See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
#
# #######################################################################
#
# Test Python API generated from templates.yaml.
#
from __future__ import print_function

import unittest
import templates

class Templates(unittest.TestCase):
    """Test templates problem"""
     
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

    def test_vector_int(self):
        v1 = templates.std.vector_int()
        self.assertIsInstance(v1, templates.std.vector_int)

        v1.push_back(1)

        ivalue = v1.at(0)
        self.assertEqual(ivalue, 1)

    def test_vector_double(self):
        v1 = templates.std.vector_double()
        self.assertIsInstance(v1, templates.std.vector_double)

        v1.push_back(1.5)

        ivalue = v1.at(0)
        self.assertEqual(ivalue, 1.5)

    def test_function_templates(self):
        templates.FunctionTU(1, 2)
        templates.FunctionTU(1.2, 2.2)
        # call function_tu(w1, w2)

        rv_int = templates.UseImplWorker_internal_ImplWorker1()
        self.assertEqual(1, rv_int)

        rv_int = templates.UseImplWorker_internal_ImplWorker2()
        self.assertEqual(2, rv_int)
        

# creating a new test suite
newSuite = unittest.TestSuite()
 
# adding a test case
newSuite.addTest(unittest.makeSuite(Templates))

if __name__ == "__main__":
    unittest.main()
