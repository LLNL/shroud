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

from shroud import declast

import unittest

class CheckParse(unittest.TestCase):
#    maxDiff = None

    # attr
    def xtest_attr01(self):
        r = parse_decl.x("+intent").attr()
        self.assertEqual(r, ('intent', True))

    def xtest_attr02(self):
        r = parse_decl.x("+intent=in").attr()
        self.assertEqual(r, ('intent', 'in'))

    def xtest_attr03(self):
        r = parse_decl.x("+intent()").attr()
        self.assertEqual(r, ('intent', ''))

    def xtest_attr04(self):
        r = parse_decl.x("+intent(in)").attr()
        self.assertEqual(r, ('intent', 'in'))

    def xtest_attr05(self):
        r = parse_decl.x('+name="abcd"').attr()
        self.assertEqual(r, ('name', 'abcd'))

    def xtest_attr06(self):
        r = parse_decl.x("+name='def'").attr()
        self.assertEqual(r, ('name', 'def'))

    def xtest_attr07(self):
        r = parse_decl.x("+ii=12").attr()
        self.assertEqual(r, ('ii', 12))

    def xtest_attr08(self):
        r = parse_decl.x("+d1=-12.0").attr()
        self.assertEqual(r, ('d1', -12.0))

    def xtest_attr09(self):
        r = parse_decl.x("+d2=11.3e-10").attr()
        self.assertEqual(r, ('d2', 1.13e-09))

    def xtest_attr10(self):
        r = parse_decl.x("+d3=11e10").attr()
        self.assertEqual(r, ('d3', 110000000000.0))

    def xtest_attr11(self):
        r = parse_decl.x("+dimension").attr()
        self.assertEqual(r, ('dimension', True))

    def xtest_attr12(self):
        r = parse_decl.x("+dimension(*)").attr()
        self.assertEqual(r, ('dimension', '*'))

    def xtest_attr13(self):
        r = parse_decl.x("+dimension(len)").attr()
        self.assertEqual(r, ('dimension', 'len'))

    # declarator
    def xtest_declarator01(self):
        r = parse_decl("int arg")
        self.assertEqual(r, {
            'type': 'int',
            'attrs': {},
            'name': 'arg'
        })
        self.assertEqual(parse_decl.str_declarator(r), "int arg")

    def xtest_declarator02(self):
        r = parse_decl.x("const int arg").declarator()
        self.assertEqual(r, {
            'type': 'int',
            'attrs': {'const': True},
            'name': 'arg'
        })
        self.assertEqual(parse_decl.str_declarator(r), "const int arg")

    def xtest_declarator03(self):
        r = parse_decl.x("badtype arg").declarator()
        self.assertEqual(r, {
            'type': 'badtype',
            'attrs': {},
            'name': 'arg'
        })
        self.assertEqual(parse_decl.str_declarator(r), "badtype arg")

    def xtest_declarator04(self):
        r = parse_decl.x("std::vector<int> &arg").declarator()
        self.assertEqual(r, {
            'type': 'std::vector',
            'attrs': {
               'reference': True,
               'template': 'int'
            },
            'name': 'arg'
            })
        self.assertEqual(parse_decl.str_declarator(r), "std::vector<int> &arg")

    def xtest_declarator05(self):
        r = parse_decl.x("std::vector<std::string> arg").declarator()
        self.assertEqual(r, {
            'type': 'std::vector',
            'attrs': {
               'template': 'std::string'
            },
            'name': 'arg'
            })
        self.assertEqual(parse_decl.str_declarator(r), "std::vector<std::string> arg")

    # parameter_list
    def xtest_parameter_list01(self):
        r = parse_decl.x('int arg').parameter_list()
        self.assertEqual(r, [{
            'type': 'int',
            'attrs': {},
            'name': 'arg'
        }])

    def xtest_parameter_list02(self):
        r = parse_decl.x('int *arg').parameter_list()
        self.assertEqual(r,[{
            'type': 'int',
            'attrs': {'ptr': True},
            'name': 'arg'
        }])

    def xtest_parameter_list03(self):
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

    def xtest_parameter_list04(self):
        r = parse_decl.x('int arg +in').parameter_list()
        self.assertEqual(r,  [{
            'type': 'int',
            'attrs': {'in': True},
            'name': 'arg'
        }])

    def xtest_parameter_list05(self):
        r = parse_decl.x('int arg +in +value').parameter_list()
        self.assertEqual(r,[{
            'type': 'int',
            'attrs': {'value': True, 'in': True},
            'name': 'arg'
        }])

    def xtest_parameter_list06(self):
        r = parse_decl.x('const string& getName').parameter_list()
        self.assertEqual(r,[{
            'type': 'string',
            'attrs': {'const': True, 'reference': True},
            'name': 'getName'
        }])

    def xtest_parameter_list07(self):
        r = parse_decl.x('std::string getName').parameter_list()
        self.assertEqual(r,[{
            'type': 'std::string',
            'attrs': {},
            'name': 'getName'
        }])

    # argument_list
    def xtest_argument_list01(self):
        r = parse_decl.x("()").argument_list()
        self.assertEqual(r, [])

    def xtest_argument_list02(self):
        r = parse_decl.x("(int arg1)").argument_list()
        self.assertEqual(r, [
            {
                'type': 'int',
                'attrs': {},
                'name': 'arg1'
            }
        ])

    def xtest_argument_list03(self):
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

    def xtest_argument_list04(self):
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
        """Simple declaration"""
        r = declast.check_decl("void foo")

        s = r.gen_decl()
        self.assertEqual("void foo", s)

        self.assertEqual(r.to_dict(),{
            'args': [],
            'attrs': {},
            'result': {
                'attrs': {},
                'name': 'foo',
                'type': 'void',
            }
        })
        self.assertEqual(r._to_dict(),{
            "attrs": {}, 
            "const": False, 
            "declarator": {
                "name": "foo", 
                "pointer": []
            }, 
            "func_const": False, 
            "specifier": [
                "void"
            ], 
            "storage": []
        })
        self.assertEqual("foo", r.get_name())

    def test_decl02(self):
        """Simple declaration with attribute"""
        r = declast.check_decl("void foo +alias(junk)")

        s = r.gen_decl()
        self.assertEqual("void foo+alias(junk)", s)

        self.assertEqual(r.to_dict(), {
            'args': [],
            'attrs': {},
            'result': {
                'attrs': {'alias': 'junk'},
                'name': 'foo',
                'type': 'void',
            }
        })
        self.assertEqual(r._to_dict(),{
            "attrs": {
                "alias": "junk"
            }, 
            "const": False, 
            "declarator": {
                "name": "foo", 
                "pointer": []
            }, 
            "func_const": False, 
            "specifier": [
                "void"
            ], 
            "storage": []
        })
        self.assertEqual("foo", r.get_name())

    def test_decl03(self):
        """Empty parameter list"""
        r = declast.check_decl("void foo()")

        s = r.gen_decl()
        self.assertEqual("void foo()", s)

        self.assertEqual(r.to_dict(), {
            'args': [],
            'attrs': {},
            'result': {
                'attrs': {},
                'name': 'foo',
                'type': 'void',
            }
        })
        self.assertEqual(r._to_dict(),{
            "args": [], 
            "attrs": {}, 
            "const": False, 
            "declarator": {
                "name": "foo", 
                "pointer": []
            }, 
            "fattrs": {}, 
            "func_const": False, 
            "specifier": [
                "void"
            ], 
            "storage": []
        })
        self.assertEqual("foo", r.get_name())

    def test_decl04(self):
        """const method"""
        r = declast.check_decl("void foo() const")

        s = r.gen_decl()
        self.assertEqual("void foo() const", s)

        self.assertEqual(r.to_dict(),{
            'args': [],
            'attrs': {'const': True},
            'result': {
                'attrs': {},
                'name': 'foo',
                'type': 'void',
            }
        })
        self.assertEqual(r._to_dict(),{
            "args": [], 
            "attrs": {}, 
            "const": False, 
            "declarator": {
                "name": "foo", 
                "pointer": []
            }, 
            "fattrs": {}, 
            "func_const": True, 
            "specifier": [
                "void"
            ], 
            "storage": []
        })
        self.assertEqual("foo", r.get_name())

    def test_decl05(self):
        """Single argument"""
        r = declast.check_decl("void foo(int arg1)")

        s = r.gen_decl()
        self.assertEqual("void foo(int arg1)", s)

        self.assertEqual(r.to_dict(),{
            'args': [
                {
                    'attrs': {},
                    'name': 'arg1',
                    'type': 'int',
                }
            ],
            'attrs': {},
            'result': {
                'attrs': {},
                'name': 'foo',
                'type': 'void',
            }
        })
        self.assertEqual(r._to_dict(),{
            "args": [
                {
                    "attrs": {}, 
                    "const": False, 
                    "declarator": {
                        "name": "arg1", 
                        "pointer": []
                    }, 
                    "func_const": False, 
                    "specifier": [
                        "int"
                    ], 
                    "storage": []
                }
            ], 
            "attrs": {}, 
            "const": False, 
            "declarator": {
                "name": "foo", 
                "pointer": []
            }, 
            "fattrs": {}, 
            "func_const": False, 
            "specifier": [
                "void"
            ], 
            "storage": []
        })
        self.assertEqual("foo", r.get_name())

    def test_decl06(self):
        """multiple arguments"""
        r = declast.check_decl("void foo(int arg1, double arg2)")

        s = r.gen_decl()
        self.assertEqual("void foo(int arg1, double arg2)", s)

        self.assertEqual(r.to_dict(), {
            'args': [{
                'attrs': {},
                'name': 'arg1',
                'type': 'int',
            },{
                'attrs': {},
                'name': 'arg2',
                'type': 'double',
            }],
            'attrs': {},
            'result': {
                'attrs': {},
                'name': 'foo',
                'type': 'void',
            }
        })
        self.assertEqual(r._to_dict(),{
            "args": [
                {
                    "attrs": {}, 
                    "const": False, 
                    "declarator": {
                        "name": "arg1", 
                        "pointer": []
                    }, 
                    "func_const": False, 
                    "specifier": [
                        "int"
                    ], 
                    "storage": []
                }, 
                {
                    "attrs": {}, 
                    "const": False, 
                    "declarator": {
                        "name": "arg2", 
                        "pointer": []
                    }, 
                    "func_const": False, 
                    "specifier": [
                        "double"
                    ], 
                    "storage": []
                }
            ], 
            "attrs": {}, 
            "const": False, 
            "declarator": {
                "name": "foo", 
                "pointer": []
            }, 
            "fattrs": {}, 
            "func_const": False, 
            "specifier": [
                "void"
            ], 
            "storage": []
        })
        self.assertEqual("foo", r.get_name())

    def test_decl07(self):
        """Return string"""
        r = declast.check_decl("const std::string& getName() const")

        s = r.gen_decl()
        self.assertEqual("const std::string &getName() const", s)

        self.assertEqual(r.to_dict(), {
            'args': [],
            'attrs': {'const': True},
            'result': {
                'attrs': {'const': True, 'reference': True},
                'name': 'getName',
                'type': 'std::string',
            }
        })
        self.assertEqual(r._to_dict(),{
            "args": [], 
            "attrs": {}, 
            "const": True, 
            "declarator": {
                "name": "getName", 
                "pointer": [
                    {
                        "const": False, 
                        "ptr": "&"
                    }
                ]
            }, 
            "fattrs": {}, 
            "func_const": True, 
            "specifier": [
                "std::string"
            ], 
            "storage": []
        })
        self.assertEqual("getName", r.get_name())

    def test_decl08(self):
        """Test attributes.
        """
        r = declast.check_decl("const void foo+attr1(30)+len=30("
                               "int arg1+in, double arg2+out)"
                               "+attr2(True)" )

        s = r.gen_decl()
        self.assertEqual("const void foo+attr1(30)+len(30)("
                         "int arg1+in, double arg2+out)"
                         "+attr2(True)", s)

        self.assertEqual(r.to_dict(), {
            'args': [
                {
                    'attrs': {'in': True},
                    'name': 'arg1',
                    'type': 'int',
                },{
                    'attrs': {'out': True},
                    'name': 'arg2',
                    'type': 'double',
                }
            ],
            'attrs': {
                'attr2' : 'True',
            },
            'result':
            {
                'attrs': {
                    'attr1': '30',
                    'const': True,
                    'len': 30,
                },
                'name': 'foo',
                'type': 'void',
            }
        })
        self.assertEqual(r._to_dict(),{
            "args": [
                {
                    "attrs": {
                        "in": True
                    }, 
                    "const": False, 
                    "declarator": {
                        "name": "arg1", 
                        "pointer": []
                    }, 
                    "func_const": False, 
                    "specifier": [
                        "int"
                    ], 
                    "storage": []
                }, 
                {
                    "attrs": {
                        "out": True
                    }, 
                    "const": False, 
                    "declarator": {
                        "name": "arg2", 
                        "pointer": []
                    }, 
                    "func_const": False, 
                    "specifier": [
                        "double"
                    ], 
                    "storage": []
                }
            ], 
            "attrs": {
                "attr1": "30", 
                "len": 30
            }, 
            "const": True, 
            "declarator": {
                "name": "foo", 
                "pointer": []
            }, 
            "fattrs": {
                "attr2": "True"
            }, 
            "func_const": False, 
            "specifier": [
                "void"
            ], 
            "storage": []
        })
        self.assertEqual("foo", r.get_name())

    def test_decl09(self):
        """Test constructor
        The type and varialbe have the same name.
        """
        r = declast.check_decl("Class1 *Class1()  +constructor",current_class='Class1')

        s = r.gen_decl()
        self.assertEqual("Class1 *Class1()+constructor", s)

        self.assertEqual(r.to_dict(),  {
            "args": [], 
            "attrs": {
                "constructor": True
            }, 
            "result": {
                "attrs": {
                    "ptr": True
                }, 
                "name": "Class1", 
                "type": "Class1"
            }
        })
        self.assertEqual(r._to_dict(),{
            "args": [], 
            "attrs": {}, 
            "const": False, 
            "declarator": {
                "name": "Class1", 
                "pointer": [
                    {
                        "const": False, 
                        "ptr": "*"
                    }
                ]
            }, 
            "fattrs": {
                "constructor": True
            }, 
            "func_const": False, 
            "specifier": [
                "Class1"
            ], 
            "storage": []
        })
        self.assertEqual("Class1", r.get_name())

    def test_decl10(self):
        """Test default arguments
        """
        r = declast.check_decl("void name(int arg1 = 0, "
                               "double arg2 = 0.0,"
                               "std::string arg3 = \"name\","
                               "bool arg4 = true)")

        s = r.gen_decl()
        self.assertEqual("void name(int arg1=0, "
                         "double arg2=0.0, "
                         "std::string arg3=\"name\", "
                         "bool arg4=true)", s)

        self.assertEqual(r.to_dict(),  {
            "args": [
                {
                    "attrs": {
                        "default": 0
                    }, 
                    "name": "arg1", 
                    "type": "int"
                }, 
                {
                    "attrs": {
                        "default": 0.0
                    }, 
                    "name": "arg2", 
                    "type": "double"
                }, 
                {
                    "attrs": {
                        "default": '"name"'
                    }, 
                    "name": "arg3", 
                    "type": "std::string"
                },
                {
                    "attrs": {
                        "default": "true"
                    }, 
                    "name": "arg4", 
                    "type": "bool"
                }
            ], 
            "attrs": {}, 
            "result": {
                "attrs": {}, 
                "name": "name", 
                "type": "void"
            }
        })
        self.assertEqual(r._to_dict(),{
            "args": [
                {
                    "attrs": {}, 
                    "const": False, 
                    "declarator": {
                        "name": "arg1", 
                        "pointer": []
                    }, 
                    "func_const": False, 
                    "init": 0,
                    "specifier": [
                        "int"
                    ], 
                    "storage": []
                }, 
                {
                    "attrs": {}, 
                    "const": False, 
                    "declarator": {
                        "name": "arg2", 
                        "pointer": []
                    }, 
                    "func_const": False, 
                    "init": 0.0,
                    "specifier": [
                        "double"
                    ], 
                    "storage": []
                }, 
                {
                    "attrs": {}, 
                    "const": False, 
                    "declarator": {
                        "name": "arg3", 
                        "pointer": []
                    }, 
                    "func_const": False,
                    "init": '"name"',
                    "specifier": [
                        "std::string"
                    ], 
                    "storage": []
                }, 
                {
                    "attrs": {}, 
                    "const": False, 
                    "declarator": {
                        "name": "arg4", 
                        "pointer": []
                    }, 
                    "init": "true",
                    "func_const": False, 
                    "specifier": [
                        "bool"
                    ], 
                    "storage": []
                }
            ], 
            "attrs": {}, 
            "const": False, 
            "declarator": {
                "name": "name", 
                "pointer": []
            }, 
            "fattrs": {}, 
            "func_const": False, 
            "specifier": [
                "void"
            ], 
            "storage": []
        })
        self.assertEqual("name", r.get_name())

    def test_decl11(self):
        """Test template_types
        """
        r = declast.check_decl("void decl11(ArgType arg)", template_types=['ArgType'])

        s = r.gen_decl()
        self.assertEqual("void decl11(ArgType arg)", s)

        self.assertEqual(r.to_dict(),  {
            "args": [
                {
                    "attrs": {}, 
                    "name": "arg", 
                    "type": "ArgType"
                }
            ], 
            "attrs": {}, 
            "result": {
                "attrs": {}, 
                "name": "decl11", 
                "type": "void"
            }
        })
        self.assertEqual(r._to_dict(),{
            "args": [
                {
                    "attrs": {}, 
                    "const": False, 
                    "declarator": {
                        "name": "arg", 
                        "pointer": []
                    }, 
                    "func_const": False, 
                    "specifier": [
                        "ArgType"
                    ], 
                    "storage": []
                }
            ], 
            "attrs": {}, 
            "const": False, 
            "declarator": {
                "name": "decl11", 
                "pointer": []
            }, 
            "fattrs": {}, 
            "func_const": False, 
            "specifier": [
                "void"
            ], 
            "storage": []
        })
        self.assertEqual("decl11", r.get_name())
                         
    def test_decl12(self):
        """Test templates
        Test std::string and string types.
        """
        r = declast.check_decl("void decl12(std::vector<std::string> arg1, string arg2)")

        s = r.gen_decl()
        self.assertEqual("void decl12(std::vector<std::string> arg1, string arg2)", s)

        self.assertEqual(r.to_dict(),  {
            "args": [
                {
                    "attrs": {
                        "template": "std::string"
                    }, 
                    "name": "arg1", 
                    "type": "std::vector"
                }, 
                {
                    "attrs": {}, 
                    "name": "arg2", 
                    "type": "string"
                }
            ], 
            "attrs": {}, 
            "result": {
                "attrs": {}, 
                "name": "decl12",
                "type": "void"
            }
        })
        self.assertEqual(r._to_dict(),{
            "args": [
                {
                    "attrs": {
                        "template": "std::string"
                    }, 
                    "const": False, 
                    "declarator": {
                        "name": "arg1", 
                        "pointer": []
                    }, 
                    "func_const": False, 
                    "specifier": [
                        "std::vector"
                    ], 
                    "storage": []
                }, 
                {
                    "attrs": {}, 
                    "const": False, 
                    "declarator": {
                        "name": "arg2", 
                        "pointer": []
                    }, 
                    "func_const": False, 
                    "specifier": [
                        "string"
                    ], 
                    "storage": []
                }
            ], 
            "attrs": {}, 
            "const": False, 
            "declarator": {
                "name": "decl12", 
                "pointer": []
            }, 
            "fattrs": {}, 
            "func_const": False, 
            "specifier": [
                "void"
            ], 
            "storage": []
        })
        self.assertEqual("decl12", r.get_name())

    def test_asarg(self):
        r = declast.check_decl("const std::string& getName() const")

        s = r.gen_decl()
        self.assertEqual("const std::string &getName() const", s)

        r.result_as_arg('output')
        s = r.gen_decl()
        self.assertEqual("void getName(std::string &output) const", s)
                         
    def test_thisarg01(self):
        """Create an argument for const this"""
        r = declast.create_this_arg('self', 'Class1', const=True)
        s = r.gen_decl()
        self.assertEqual("const Class1 *self", s)

    def test_thisarg02(self):
        """Create an argument for this"""
        r = declast.create_this_arg('self', 'Class1', const=False)
        s = r.gen_decl()
        self.assertEqual("Class1 *self", s)

                         
if __name__ == '__main__':
    unittest.main()
