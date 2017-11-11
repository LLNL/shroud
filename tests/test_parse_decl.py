"""
Copyright (c) 2017, Lawrence Livermore National Security, LLC. 
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
Parse C++ declarations.
"""
from __future__ import print_function

from shroud import parse_decl

import unittest

class CheckDeclCase(unittest.TestCase):

    # qualifier
    def test_qualifier01(self):
        r = parse_decl.x("const").qualifier()
        self.assertEqual(r, [('const', True)])

    def test_qualifier02(self):
        r = parse_decl.x("const").qualifier()
        self.assertEqual(r, [('const', True)])

    # pointer
    def test_pointer01(self):
        r = parse_decl.x("*").pointer()
        self.assertEqual(r, [('ptr', True)])

    def test_pointer02(self):
        r = parse_decl.x("&").pointer()
        self.assertEqual(r, [('reference', True)])

    def test_pointer03(self):
        r = parse_decl.x("").pointer()
        self.assertEqual(r, [])

    # templat
    def test_template01(self):
        r = parse_decl.x("<int>").template()
        self.assertEqual(r, [('template', 'int')])

    def test_template02(self):
        r = parse_decl.x("< int >").template()
        self.assertEqual(r, [('template', 'int')])

    # attr
    def test_attr01(self):
        r = parse_decl.x("+intent").attr()
        self.assertEqual(r, ('intent', True))

    def test_attr02(self):
        r = parse_decl.x("+intent=in").attr()
        self.assertEqual(r, ('intent', 'in'))

    def test_attr03(self):
        r = parse_decl.x("+intent()").attr()
        self.assertEqual(r, ('intent', ''))

    def test_attr04(self):
        r = parse_decl.x("+intent(in)").attr()
        self.assertEqual(r, ('intent', 'in'))

    def test_attr05(self):
        r = parse_decl.x('+name="abcd"').attr()
        self.assertEqual(r, ('name', 'abcd'))

    def test_attr06(self):
        r = parse_decl.x("+name='def'").attr()
        self.assertEqual(r, ('name', 'def'))

    def test_attr07(self):
        r = parse_decl.x("+ii=12").attr()
        self.assertEqual(r, ('ii', 12))

    def test_attr08(self):
        r = parse_decl.x("+d1=-12.0").attr()
        self.assertEqual(r, ('d1', -12.0))

    def test_attr09(self):
        r = parse_decl.x("+d2=11.3e-10").attr()
        self.assertEqual(r, ('d2', 1.13e-09))

    def test_attr10(self):
        r = parse_decl.x("+d3=11e10").attr()
        self.assertEqual(r, ('d3', 110000000000.0))

    def test_attr11(self):
        r = parse_decl.x("+dimension").attr()
        self.assertEqual(r, ('dimension', True))

    def test_attr12(self):
        r = parse_decl.x("+dimension(*)").attr()
        self.assertEqual(r, ('dimension', '*'))

    def test_attr13(self):
        r = parse_decl.x("+dimension(len)").attr()
        self.assertEqual(r, ('dimension', 'len'))

    # declarator
    def test_declarator01(self):
        r = parse_decl.x("int arg").declarator()
        self.assertEqual(r, {
            'type': 'int',
            'attrs': {},
            'name': 'arg'
        })

    def test_declarator02(self):
        r = parse_decl.x("const int arg").declarator()
        self.assertEqual(r, {
            'type': 'int',
            'attrs': {'const': True},
            'name': 'arg'
        })

    def test_declarator03(self):
        r = parse_decl.x("badtype arg").declarator()
        self.assertEqual(r, {
            'type': 'badtype',
            'attrs': {},
            'name': 'arg'
        })

    def test_declarator04(self):
        r = parse_decl.x("std::vector<int> &arg").declarator()
        self.assertEqual(r, {
            'type': 'std::vector',
            'attrs': {
               'reference': True,
               'template': 'int'
            },
            'name': 'arg'
            })

    def test_declarator05(self):
        r = parse_decl.x("std::vector<std::string> &arg").declarator()
        self.assertEqual(r, {
            'type': 'std::vector',
            'attrs': {
               'reference': True,
               'template': 'std::string'
            },
            'name': 'arg'
            })

    # parameter_list
    def test_parameter_list01(self):
        r = parse_decl.x('int arg').parameter_list()
        self.assertEqual(r, [{
            'type': 'int',
            'attrs': {},
            'name': 'arg'
        }])

    def test_parameter_list02(self):
        r = parse_decl.x('int *arg').parameter_list()
        self.assertEqual(r,[{
            'type': 'int',
            'attrs': {'ptr': True},
            'name': 'arg'
        }])

    def test_parameter_list03(self):
        r = parse_decl.x('int arg1, double arg2').parameter_list()
        self.assertEqual(r, [{
            'type': 'int',
            'attrs': {},
            'name': 'arg1'
        },{
            'type': 'double',
            'attrs': {},
            'name': 'arg2'
        }])

    def test_parameter_list04(self):
        r = parse_decl.x('int arg +in').parameter_list()
        self.assertEqual(r,  [{
            'type': 'int',
            'attrs': {'in': True},
            'name': 'arg'
        }])

    def test_parameter_list05(self):
        r = parse_decl.x('int arg +in +value').parameter_list()
        self.assertEqual(r,[{
            'type': 'int',
            'attrs': {'value': True, 'in': True},
            'name': 'arg'
        }])

    def test_parameter_list06(self):
        r = parse_decl.x('const string& getName').parameter_list()
        self.assertEqual(r,[{
            'type': 'string',
            'attrs': {'const': True, 'reference': True},
            'name': 'getName'
        }])

    def test_parameter_list07(self):
        r = parse_decl.x('std::string getName').parameter_list()
        self.assertEqual(r,[{
            'type': 'std::string',
            'attrs': {},
            'name': 'getName'
        }])

    # argument_list
    def test_argument_list01(self):
        r = parse_decl.x("()").argument_list()
        self.assertEqual(r, [])

    def test_argument_list02(self):
        r = parse_decl.x("(int arg1)").argument_list()
        self.assertEqual(r, [
            {
                'type': 'int',
                'attrs': {},
                'name': 'arg1'
            }
        ])

    def test_argument_list03(self):
        r = parse_decl.x("(int arg1, double arg2)").argument_list()
        self.assertEqual(r, [
            {
                'type': 'int',
                'attrs': {},
                'name': 'arg1'
            },{
                'type': 'double',
                'attrs': {},
                'name': 'arg2'
            }
        ])

    def test_argument_list04(self):
        r = parse_decl.x("(int arg1, double arg2 = 0.0)").argument_list()
        self.assertEqual(r,  [
            {
                'type': 'int',
                'attrs': {},
                'name': 'arg1'
            },{
                'type': 'double',
                'attrs': {'default': 0.0},
                'name': 'arg2'
            }
        ])

    # decl
    def test_decl01(self):
        r = parse_decl.check_decl("void foo")
        self.assertEqual(r,{
            'args': [],
            'attrs': {},
            'result': {
                'type': 'void',
                'attrs': {},
                'name': 'foo'
            }
        })

    def test_decl02(self):
        r = parse_decl.check_decl("void foo +alias=junk")
        self.assertEqual(r, {
            'args': [],
            'attrs': {},
            'result': {
                'type': 'void',
                'attrs': {'alias': 'junk'},
                'name': 'foo'
            }
        })

    def test_decl03(self):
        r = parse_decl.check_decl("void foo()")
        self.assertEqual(r, {
            'args': [],
            'attrs': {},
            'result': {
                'type': 'void',
                'attrs': {},
                'name': 'foo'
            }
        })

    def test_decl04(self):
        r = parse_decl.check_decl("void foo() const")
        self.assertEqual(r,{
            'args': [],
            'attrs': {'const': True},
            'result': {
                'type': 'void',
                'attrs': {},
                'name': 'foo'
            }
        })

    def test_decl05(self):
        r = parse_decl.check_decl("void foo(int arg1)")
        self.assertEqual(r,{
            'args': [
                {
                    'type': 'int',
                    'attrs': {},
                    'name': 'arg1'
                }
            ],
            'attrs': {},
            'result': {
                'type': 'void',
                'attrs': {},
                'name': 'foo'
            }
        })

    def test_decl06(self):
        r = parse_decl.check_decl("void foo(int arg1, double arg2)")
        self.assertEqual(r, {
            'args': [{
                'type': 'int',
                'attrs': {},
                'name': 'arg1'
            },{
                'type': 'double',
                'attrs': {},
                'name': 'arg2'
            }],
            'attrs': {},
            'result': {
                'type': 'void',
                'attrs': {},
                'name': 'foo'
            }
        })

    def test_decl07(self):
        r = parse_decl.check_decl("const std::string& getName() const")
        self.assertEqual(r, {
            'args': [],
            'attrs': {'const': True},
            'result': {
                'type': 'std::string',
                'attrs': {'const': True, 'reference': True},
                'name': 'getName'
            }
        })

    def test_decl08(self):
        r = parse_decl.check_decl("const void foo("
                                  "int arg1+in, double arg2+out)")
        self.assertEqual(r, {
            'args': [
                {
                    'type': 'int',
                    'attrs': {'in': True},
                    'name': 'arg1'
                },{
                    'type': 'double',
                    'attrs': {'out': True},
                    'name': 'arg2'
                }
            ],
            'attrs': {},
            'result':
            {
                'type': 'void',
                'attrs': {'const': True},
                'name': 'foo'
            }
        })

    def test_decl09(self):
        r = parse_decl.check_decl("void new() + constructor")
        self.assertEqual(r, {
            'args': [],
            'attrs': {'constructor': True},
            'result': {
                'type': 'void',
                'attrs': {},
                'name': 'new'
            }
        })


if __name__ == '__main__':
    unittest.main()
