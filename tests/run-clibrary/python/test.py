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
# test the tutorial module
#

import unittest
import clibrary


class NotTrue:
    """Test bool arguments errors"""
    def __bool__(self):
        raise NotImplementedError
 
class Tutorial(unittest.TestCase):
    """Test tutorial problem"""
     
    def XXsetUp(self):
        """ Setting up for the test """
        print "FooTest:setUp_:begin"
        ## do something...
        print "FooTest:setUp_:end"
     
    def XXtearDown(self):
        """Cleaning up after the test"""
        print "FooTest:tearDown_:begin"
        ## do something...
        print "FooTest:tearDown_:end"
     
    def testfunction2(self):
        self.assertEqual(5.0, clibrary.Function2(1.0, 4))

    def testfunction3(self):
        self.assertEqual(True, clibrary.Function3(False))

    def testfunction3b(self):
        self.assertEqual((False, False), clibrary.Function3b(True, True))

    def testfunction4a(self):
        self.assertEqual('dogcat', clibrary.Function4a('dog', 'cat'))

    def testintargs(self):
        self.assertEqual((1, 2), clibrary.intargs(1, 2))

    def testsum(self):
        self.assertEqual(15, clibrary.Sum([1, 2, 3, 4, 5]))


# creating a new test suite
newSuite = unittest.TestSuite()
 
# adding a test case
newSuite.addTest(unittest.makeSuite(Tutorial))

if __name__ == "__main__":
    unittest.main()
