# Copyright Shroud Project Developers. See LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
# #######################################################################
#
# Test Python API generated from strings.yaml.
#

import unittest
import strings

# value from strings.cpp
static_char = 'bird'
static_str  = 'dog'

strs_array = ["apple", "pear", "peach", "cherry"]

class NotTrue:
    """Test bool arguments errors"""
    def __bool__(self):
        raise NotImplementedError
 
class Strings(unittest.TestCase):
    """Test tutorial problem"""
     
    def setUp(self):
        """ Setting up for the test """
        strings.init_test()
#        print("FooTest:setUp_:begin")
#        ## do something...
#        print("FooTest:setUp_:end")
     
    def XXtearDown(self):
        """Cleaning up after the test"""
        print("FooTest:tearDown_:begin")
        ## do something...
        print("FooTest:tearDown_:end")
     
    def testgetConstString(self):
        """return std::string"""
        self.assertEqual(static_str, strings.getConstStringLen())
        self.assertEqual('getConstStringAlloc', strings.getConstStringAlloc())

        self.assertEqual(static_str, strings.getConstStringAsArg())

    def testgetConstStringRef(self):
        """return std::string reference"""
        # The variations are useful for the Fortran API,
        # but really no difference in the Python API.
        self.assertEqual(static_str, strings.getConstStringRefLen())
        self.assertEqual('', strings.getConstStringRefLenEmpty())
        self.assertEqual(static_str, strings.getConstStringRefAlloc())

        self.assertEqual(static_str, strings.getConstStringRefAsArg())

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

    def testacceptStringInstance(self):
        s = "acceptStringInstance"
        nlen = strings.acceptStringInstance(s)
        self.assertEqual(len(s), nlen)
        
    def testreturnStrings(self):
        self.assertEqual(('up', 'down'), strings.returnStrings())

    #- decl: void acceptStringInstance(std::string arg1)

    def testCpassChar(self):
        strings.CpassChar('w')

    def testCreturnChar(self):
        self.assertEqual('w', strings.CreturnChar())


unittest.TestLoader().loadTestsFromTestCase(Strings)

if __name__ == "__main__":
    unittest.main()
