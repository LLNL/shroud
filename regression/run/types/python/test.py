# Copyright (c) 2017-2020, Lawrence Livermore National Security, LLC and
# other Shroud Project Developers.
# See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
#
# #######################################################################
#
# Test Python API generated from types.yaml.
#
from __future__ import print_function

import unittest
import shtypes

class NotTrue:
    """Test bool arguments errors"""
    def __bool__(self):
        raise NotImplementedError
 
class Types(unittest.TestCase):
    """Test types problem"""
     
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

    def test_short(self):
        rv_short = shtypes.short_func(1)
        self.assertEqual(rv_short, 1)

    def test_int(self):
        rv_int = shtypes.int_func(1)
        self.assertEqual(rv_int, 1)

    def test_long(self):
        rv_long = shtypes.long_func(1)
        self.assertEqual(rv_long, 1)

    def test_long_long(self):
        rv_long_long = shtypes.long_long_func(1)
        self.assertEqual(rv_long_long, 1)

#    def test_(self):
#        ! explicit int
#        rv_short = shtypes.short_int_func(1)
#        self.assertEqual(rv_short, 1)

#    def test_(self):
#        rv_long = shtypes.long_int_func(1)
#        self.assertEqual(rv_long, 1)

#    def test_(self):
#        rv_long_long = shtypes.long_long_int_func(1)
#        self.assertEqual(rv_long_long, 1)

    def test_unsigned(self):
        rv_int = shtypes.unsigned_func(1)
        self.assertEqual(rv_int, 1)

    def test_ushort(self):
        rv_short = shtypes.ushort_func(1)
        self.assertEqual(rv_short, 1)

    def test_uint(self):
        rv_int = shtypes.uint_func(1)
        self.assertEqual(rv_int, 1)

    def test_ulong(self):
        rv_long = shtypes.ulong_func(1)
        self.assertEqual(rv_long, 1)

    def test_ulong_long(self):
        rv_long_long = shtypes.ulong_long_func(1)
        self.assertEqual(rv_long_long, 1)

#    def test_(self):
#        ! implied int
#        rv_long = shtypes.ulong_int_func(1)
#        self.assertEqual(rv_long, 1)

#    def test_(self):
#        ! test negative number, C treats as large unsigned number.
#        rv_int = shtypes.-1_C_INT
#        rv_int = shtypes.uint_func(rv_int)
#        self.assertEqual(rv_int, -1)

    def test_int8(self):
        rv_int8 = shtypes.int8_func(1)
        self.assertEqual(rv_int8, 1)

    def test_int16(self):
        rv_int16 = shtypes.int16_func(1)
        self.assertEqual(rv_int16, 1)

    def test_int32(self):
        rv_int32 = shtypes.int32_func(1)
        self.assertEqual(rv_int32, 1)

    def test_int64(self):
        rv_int64 = shtypes.int64_func(1)
        self.assertEqual(rv_int64, 1)

    # unsigned
    def test_uint8(self):
        rv_int8 = shtypes.uint8_func(1)
        self.assertEqual(rv_int8, 1)

    def test_uint16(self):
        rv_int16 = shtypes.uint16_func(1)
        self.assertEqual(rv_int16, 1)

    def test_uint32(self):
        rv_int32 = shtypes.uint32_func(1)
        self.assertEqual(rv_int32, 1)

    def XXXtest_uint64(self):
        rv_int64 = shtypes.uint64_func(1)
        self.assertEqual(rv_int64, 1)

    def test_size_func(self):
        rv = shtypes.size_func(1)
        self.assertEqual(1, rv)

# creating a new test suite
newSuite = unittest.TestSuite()
 
# adding a test case
newSuite.addTest(unittest.makeSuite(Types))

if __name__ == "__main__":
    unittest.main()
