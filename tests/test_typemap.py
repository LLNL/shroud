"""
Copyright (c) 2018, Lawrence Livermore National Security, LLC. 
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

"""
from __future__ import print_function

from shroud import typemap

import unittest

class NameSpace(unittest.TestCase):

    def test_ns1(self):
        glb = typemap.Namespace(None)
        self.assertEqual('', glb.scope)

        # typedef foo;
        # foo var;
        glb.add_typedef('foo')
        typ = glb.qualified_lookup('foo')
        self.assertTrue(typ)
        typ = glb.unqualified_lookup('foo')
        self.assertTrue(typ)

        std = typemap.create_std_namespace(glb)
        self.assertEqual('std::', std.scope)

        # std::foo
        typ = std.qualified_lookup('foo')
        self.assertFalse(typ)

        # namespace std { foo var; }
        typ = std.unqualified_lookup('foo')
        self.assertTrue(typ)
        # namespace std { string var; }
        typ = std.unqualified_lookup('string')
        self.assertTrue(typ)
        # string var;
        typ = glb.unqualified_lookup('string')
        self.assertFalse(typ)

        # using namespace std;
        # string var;
        glb.using_directive('std')
        typ = glb.unqualified_lookup('string')
        self.assertTrue(typ)

        glb.using_directive('std')
        self.assertEqual(1, len(glb.using))
