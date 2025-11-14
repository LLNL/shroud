# Copyright Shroud Project Developers. See LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
# #######################################################################
#
# Test Python API generated from strings.yaml.
#

import unittest

import char

# value from char.c
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
        char.init_test()
#        print("FooTest:setUp_:begin")
#        ## do something...
#        print("FooTest:setUp_:end")
     
    def XXtearDown(self):
        """Cleaning up after the test"""
        print("FooTest:tearDown_:begin")
        ## do something...
        print("FooTest:tearDown_:end")
     
    #- decl: void acceptStringInstance(std::string arg1)

    def testCpassChar(self):
        char.CpassChar('w')

    def testCreturnChar(self):
        self.assertEqual('w', char.CreturnChar())

    def test_acceptCharArrayIn(self):
        n = char.acceptCharArrayIn(["dog", "cat", "monkey"])
        self.assertEqual(len("dog"), n)

    def test_fetchCharPtrLibrary(self):
        outstr = char.fetchCharPtrLibrary()
        self.assertEqual("static_char_array", outstr)

    def test_fetchCharPtrLibraryNULL(self):
        # Test when outstr is NULL
        irv, outstr = char.fetchCharPtrLibraryNULL()
        self.assertEqual(0, irv)
        self.assertIs(None, outstr)

unittest.TestLoader().loadTestsFromTestCase(Strings)

if __name__ == "__main__":
    unittest.main()
