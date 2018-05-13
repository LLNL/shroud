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
import ownership

class Ownership(unittest.TestCase):
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

    def testReturnIntPtr(self):
        "Return pointer to int scalar"
        rv = ownership.ReturnIntPtr()
        self.assertIsInstance(rv, np.ndarray)
        self.assertEqual('int32', rv.dtype.name)
        self.assertEqual(1, rv.size)
        self.assertEqual(1, rv)

    def testReturnIntPtrScalr(self):
        "Return pointer as int scalar"
        rv = ownership.ReturnIntPtrScalar()
        self.assertIsInstance(rv, int)
        self.assertEqual(10, rv)

    def testReturnIntPtrDim(self):
        "Return pointer to int array"
        rv = ownership.ReturnIntPtrDim()
        self.assertIsInstance(rv, np.ndarray)
        self.assertEqual('int32', rv.dtype.name)
        self.assertEqual(7, rv.size)
        self.assertTrue(all(np.equal(rv, [1,2,3,4,5,6,7])))

    def testReturnIntPtrDimNew(self):
        "Return pointer to a new int array"
        rv = ownership.ReturnIntPtrDimNew()
        self.assertIsInstance(rv, np.ndarray)
        self.assertEqual('int32', rv.dtype.name)
        self.assertEqual(5, rv.size)
        self.assertTrue(all(np.equal(rv, [0,1,2,3,4])))


# creating a new test suite
newSuite = unittest.TestSuite()
 
# adding a test case
newSuite.addTest(unittest.makeSuite(Ownership))

if __name__ == "__main__":
    unittest.main()
