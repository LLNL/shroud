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
# test the strings module
#
from __future__ import print_function

import unittest
import strings

# value from strings.cpp
static_char = 'bird'
static_str  = 'dog'

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
     
    def testpassChar(self):
        strings.passChar('w')

    def testreturnChar(self):
        self.assertEqual('w', strings.returnChar())

    def testpassCharPtrInOut(self):
        """char * +intent(out)"""
        self.assertEqual('DOG', strings.passCharPtrInOut('dog'))

    def testgetChar(self):
        # The variations are useful for the Fortran API,
        # but really no difference in the Python API.
        self.assertEqual(static_char, strings.getCharPtr1())
        self.assertEqual(static_char, strings.getCharPtr2())
        self.assertEqual(static_char, strings.getCharPtr3())

    def testgetStringRef(self):
        """return std::string reference"""
        # The variations are useful for the Fortran API,
        # but really no difference in the Python API.
        self.assertEqual(static_str, strings.getString1())
        self.assertEqual(static_str, strings.getString2())
        self.assertEqual(static_str, strings.getString3())

        self.assertEqual('', strings.getString2_empty())
        self.assertEqual(static_str, strings.getStringRefAlloc())

    def testgetString5(self):
        """return std::string"""
        self.assertEqual(static_str, strings.getString5())
        self.assertEqual(static_str, strings.getString6())
        self.assertEqual('getStringAlloc', strings.getStringAlloc())

    def testgetString7(self):
        """return std::string pointer"""
        self.assertEqual('Hello', strings.getString7())

    def testacceptStringConstReference(self):
        self.assertEqual(None, strings.acceptStringConstReference('cat'))

    def testacceptStringReferenceOut(self):
        self.assertEqual('dog', strings.acceptStringReferenceOut())

    def testacceptStringReference(self):
        self.assertEqual('catdog', strings.acceptStringReference('cat'))

    def testacceptStringPointer(self):
        self.assertEqual('birddog', strings.acceptStringPointer('bird'))

    def testreturnStrings(self):
        self.assertEqual(('up', 'down'), strings.returnStrings())

    #- decl: void acceptStringInstance(std::string arg1)

    def testCpassChar(self):
        strings.CpassChar('w')

    def testCreturnChar(self):
        self.assertEqual('w', strings.CreturnChar())


# creating a new test suite
newSuite = unittest.TestSuite()
 
# adding a test case
newSuite.addTest(unittest.makeSuite(Tutorial))

if __name__ == "__main__":
    unittest.main()
