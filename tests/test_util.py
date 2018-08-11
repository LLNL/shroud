"""
Copyright (c) 2017-2018, Lawrence Livermore National Security, LLC. 
Produced at the Lawrence Livermore National Laboratory 

LLNL-CODE-738041.
All rights reserved. 

This file is part of Shroud.  For details, see
https://github.com/LLNL/shroud. Please also read shroud/LICENSE.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

* Redistributions of source code must retain the above copyright
  notice, this list of conditions and the disclaimer below.

* Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the disclaimer (as noted below)
  in the documentation and/or other materials provided with the
  distribution.

* Neither the name of the LLNS/LLNL nor the names of its contributors
  may be used to endorse or promote products derived from this
  software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL LAWRENCE
LIVERMORE NATIONAL SECURITY, LLC, THE U.S. DEPARTMENT OF ENERGY OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

########################################################################
Test utility module
"""
from __future__ import print_function

from shroud import util

import unittest


class UtilCase(unittest.TestCase):
    def test_un_camel(self):
        self.assertEqual(util.un_camel("incrementCount"), "increment_count")
        self.assertEqual(util.un_camel("local_function1"), "local_function1")
        self.assertEqual(
            util.un_camel("getHTTPResponseCode"), "get_http_response_code"
        )


class ScopeCase(unittest.TestCase):
    def setUp(self):
        self.lev0 = util.Scope(None, a=1, b=2, c=3)
        self.lev1 = util.Scope(self.lev0, x=100, y=1, z=102)

    def test_access01(self):
        # 'a' accessable from both
        self.assertEqual(self.lev0.a, 1)
        self.assertEqual(self.lev1.a, 1)

        # 'z' only accessable from lev1
        with self.assertRaises(AttributeError):
            self.lev0.z
        self.assertEqual(self.lev1.z, 102)

    def test_access02(self):
        """set and access"""
        self.lev0.c2 = 32
        self.assertEqual(self.lev0.c2, 32)

    def test_get01(self):
        self.assertEqual(self.lev1.get("a", "notfound"), 1)
        self.assertEqual(self.lev1.get("nosuch", "notfound"), "notfound")

    def test_in(self):
        self.assertIn("a", self.lev0)
        self.assertIn("a", self.lev1)

        self.assertNotIn("z", self.lev0)
        self.assertIn("z", self.lev1)

        self.assertNotIn("nosuch", self.lev1)

    def test_setdefault(self):
        lev1 = self.lev1
        self.assertNotIn("yyy", lev1)
        lev1.setdefault("yyy", "yyyvalue")
        self.assertIn("yyy", lev1)
        self.assertEqual(lev1.yyy, "yyyvalue")

    def test_update(self):
        self.assertEqual(self.lev0.a, 1)

        self.lev0.update(dict(a=100), replace=True)
        self.assertEqual(self.lev0.a, 100)


if __name__ == "__main__":
    unittest.main()
