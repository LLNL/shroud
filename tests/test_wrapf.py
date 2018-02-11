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

from shroud import ast
from shroud import util
from shroud import wrapf

import unittest

class CheckAllocatable(unittest.TestCase):
    def setUp(self):
        # Create a dictionary of parsed arguments
        self.library = ast.LibraryNode()
        node = self.library.add_function(
            'void func1('
            'int *in1 +intent(in)+dimension(:),'
            'int *in2 +intent(in)+dimension(:,:),'
            'int *out +intent(out),'
            'int flag)')

        node._fmtargs = dict(
            in1 = dict(
                fmtf = util.Scope(
                    None,
                    f_var='in1'
                ),
            ),
            in2 = dict(
                fmtf = util.Scope(
                    None,
                    f_var='in2',
                ),
            ),
            out = dict(
                fmtf = util.Scope(
                    None,
                    f_var='out',
                )
            ),
            flag = dict(
                fmtf = util.Scope(
                    None,
                    f_var='flag',
                )
            ),
        )
        self.func1 = node

    def test_errors(self):
        self.library.options.F_standard=2003
        pre_call = []

        with self.assertRaises(RuntimeError) as context:
            wrapf.attr_allocatable(
                'mold=none', self.func1, self.func1.ast.params[2], pre_call)
        self.assertTrue('does not exist' in str(context.exception))

        with self.assertRaises(RuntimeError) as context:
            wrapf.attr_allocatable(
                'mold=flag', self.func1, self.func1.ast.params[2], pre_call)
        self.assertTrue('must have dimension attribute'
                        in str(context.exception))

    def test_allocatable1d(self):
        self.library.options.F_standard=2003
        pre_call = []
        wrapf.attr_allocatable(
            'mold=in1', self.func1, self.func1.ast.params[2], pre_call)
        self.assertEqual('allocate(out(lbound(in1,1):ubound(in1,1)))',
                         pre_call[0])

        self.library.options.F_standard=2008
        pre_call = []
        wrapf.attr_allocatable(
            'mold=in1', self.func1, self.func1.ast.params[2], pre_call)
        self.assertEqual('allocate(out, mold=in1)', pre_call[0])

    def test_allocatable2d(self):
        self.library.options.F_standard=2003
        pre_call = []
        wrapf.attr_allocatable(
            'mold=in2', self.func1, self.func1.ast.params[2], pre_call)
        self.assertEqual('allocate(out(lbound(in2,1):ubound(in2,1),'
                         'lbound(in2,2):ubound(in2,2)))',
                         pre_call[0])

        self.library.options.F_standard=2008
        pre_call = []
        wrapf.attr_allocatable(
            'mold=in2', self.func1, self.func1.ast.params[2], pre_call)
        self.assertEqual('allocate(out, mold=in2)', pre_call[0])


if __name__ == '__main__':
    unittest.main()

