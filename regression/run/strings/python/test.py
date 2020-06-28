# Copyright (c) 2017-2020, Lawrence Livermore National Security, LLC and
# other Shroud Project Developers.
# See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
# #######################################################################
#
# Test Python API generated from strings.yaml.
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

    def testpassCharPtr(self):
        out = strings.passCharPtr("elephant")
        self.assertEqual("elephant", out)

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

    def testgetConstString(self):
        """return std::string"""
        self.assertEqual('getConstStringResult', strings.getConstStringResult())
        self.assertEqual(static_str, strings.getConstStringLen())
        self.assertEqual(static_str, strings.getConstStringAsArg())
        self.assertEqual('getConstStringAlloc', strings.getConstStringAlloc())

    def testgetConstStringRef(self):
        """return std::string reference"""
        # The variations are useful for the Fortran API,
        # but really no difference in the Python API.
        self.assertEqual(static_str, strings.getConstStringRefPure())
        self.assertEqual(static_str, strings.getConstStringRefLen())
        self.assertEqual(static_str, strings.getConstStringRefAsArg())

        self.assertEqual('', strings.getConstStringRefLenEmpty())
        self.assertEqual(static_str, strings.getConstStringRefAlloc())

    def testgetConstStringPtr(self):
        """return std::string pointer"""
        self.assertEqual('getConstStringPtrLen', strings.getConstStringPtrLen())
        self.assertEqual(static_str, strings.getConstStringPtrAlloc())
        self.assertEqual('getConstStringPtrOwnsAlloc',
                         strings.getConstStringPtrOwnsAlloc())
        self.assertEqual('getConstStringPtrOwnsAllocPattern',
                         strings.getConstStringPtrOwnsAllocPattern())

    def testacceptStringConstReference(self):
        self.assertEqual(None, strings.acceptStringConstReference('cat'))

    def testacceptStringReferenceOut(self):
        self.assertEqual('dog', strings.acceptStringReferenceOut())

    def testacceptStringReference(self):
        self.assertEqual('catdog', strings.acceptStringReference('cat'))

    def testacceptStringPointer(self):
        # Store in global_str.
        strings.acceptStringPointerConst('from Python')

        # Fetch from global_str.
        self.assertEqual('from Python', strings.fetchStringPointer())

        s, nlen = strings.fetchStringPointerLen()
        self.assertEqual('from Python', s)
        self.assertEqual(len(s), nlen)

        # append "dog".
        self.assertEqual('birddog', strings.acceptStringPointer('bird'))

        s, nlen = strings.acceptStringPointerLen('bird')
        self.assertEqual('birddog', s)
        self.assertEqual(len(s), nlen)

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
