# Copyright (c) 2018, Lawrence Livermore National Security, LLC. 
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
# test the pointers module
#
from __future__ import print_function

import math
import numpy as np
import unittest
import pointers


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
     
    def testintargs(self):
        self.assertEqual((1, 2), pointers.intargs(1, 2))

    def testcos_doubles(self):
        # x = np.arange(0, 2 * np.pi, 0.1)
        inarray = [ 0.0, 0.5*np.pi, np.pi, 1.5*np.pi, 2.0*np.pi ]
        outarray = [ math.cos(v) for v in inarray]
        rv = pointers.cos_doubles(inarray)
        self.assertTrue(isinstance(rv, np.ndarray))
        self.assertEqual('float64', rv.dtype.name)
        self.assertTrue(np.allclose(rv, outarray))

    def test_truncate(self):
        rv = pointers.truncate_to_int([1.2, 2.3, 3.4, 4.5])
        self.assertTrue(isinstance(rv, np.ndarray))
        self.assertEqual('int32', rv.dtype.name)
        self.assertTrue(np.equal(rv, [1, 2, 3, 4]).all())

    def test_increment(self):
        # the argument is return as the result because intent(INOUT)
        array = np.array([2,4,6,8], dtype=np.intc)  # int32
        out = pointers.increment(array)
        self.assertIs(array, out)
        self.assertTrue(isinstance(out, np.ndarray))
        self.assertEqual('int32', out.dtype.name)
        self.assertTrue(np.equal(out, [3,5,7,9]).all())

        # Call with incorrect argument type
        with self.assertRaises(ValueError) as context:
            array = np.array([2,4,6,8], dtype=np.float)
            out = pointers.increment(array)
        self.assertTrue('array must be' in str(context.exception))

    def test_get_values(self):
        # out - created NumPy array.
        nout, out = pointers.get_values()
        self.assertTrue(isinstance(out, np.ndarray))
        self.assertEqual('int32', out.dtype.name)
        self.assertTrue(np.equal(out, [1,2,3]).all())

    def test_get_values2(self):
        # out - created NumPy array.
        arg1, arg2 = pointers.get_values2()

        self.assertTrue(isinstance(arg1, np.ndarray))
        self.assertEqual('int32', arg1.dtype.name)
        self.assertTrue(np.equal(arg1, [1,2,3]).all())

        self.assertTrue(isinstance(arg2, np.ndarray))
        self.assertEqual('int32', arg2.dtype.name)
        self.assertTrue(np.equal(arg2, [11,12,13]).all())

# creating a new test suite
newSuite = unittest.TestSuite()
 
# adding a test case
newSuite.addTest(unittest.makeSuite(Tutorial))

if __name__ == "__main__":
    unittest.main()
