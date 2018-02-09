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

from shroud import util
from shroud import wrapp

import unittest

class CheckImplied(unittest.TestCase):
    def setUp(self):
        # Create a dictionary of parsed arguments
        self.fmtargs = dict(
            array = dict(
                fmtpy = util.Scope(
                    None,
                    numpy_var='SHAPy_array'
                ),
            ),
        )

    def test_errors(self):
        with self.assertRaises(RuntimeError) as context:
            wrapp.py_implied( 'bad(array)', self.fmtargs)
        self.assertTrue('Unexpected function' in str(context.exception))

        with self.assertRaises(RuntimeError) as context:
            wrapp.py_implied('size(array,n2)', self.fmtargs)
        self.assertTrue('Too many arguments' in str(context.exception))

        with self.assertRaises(RuntimeError) as context:
            wrapp.py_implied('size(array2)', self.fmtargs)
        self.assertTrue('Unknown argument' in str(context.exception))

    def test_implied1(self):
        self.assertEqual('PyArray_SIZE(SHAPy_array)',
                         wrapp.py_implied('size(array)', self.fmtargs))
        self.assertEqual('PyArray_SIZE(SHAPy_array)+2',
                         wrapp.py_implied('size(array) + 2', self.fmtargs))

    def test_expr1(self):
        self.assertEqual('size+n',
                         wrapp.py_implied('size+n', self.fmtargs))


if __name__ == '__main__':
    unittest.main()

