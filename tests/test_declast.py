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
Parse C++ declarations.
"""
from __future__ import print_function

from shroud import ast
from shroud import declast
from shroud import todict
from shroud import typemap
from shroud import util

import unittest
import copy

class CheckParse(unittest.TestCase):
    maxDiff = None

    def setUp(self):
        typemap.initialize()
#        declast.create_global_namespace()  # also done in LibraryNode

        self.library = ast.LibraryNode(library='cc')
        self.class1 = self.library.add_class('Class1')

    # types
    def test_type_int(self):
        """Test variable declarations
        Combinations of const and pointer.
        """
        r = declast.check_decl("int")
        self.assertIsNone(r.get_subprogram())
        self.assertEqual(0, r.is_pointer())
        s = r.gen_decl()
        self.assertEqual("int", s)

        r = declast.check_decl("int var1")
        self.assertIsNone(r.get_subprogram())
        self.assertEqual(0, r.is_pointer())
        s = r.gen_decl()
        self.assertEqual("int var1", s)
        s = r.bind_c()
        self.assertEqual("integer(C_INT) :: var1", s)
        s = r.gen_arg_as_fortran()
        self.assertEqual("integer(C_INT) :: var1", s)

        r = declast.check_decl("const int var1")
        self.assertIsNone(r.get_subprogram())
        self.assertEqual(0, r.is_pointer())
        s = r.gen_decl()
        self.assertEqual("const int var1", s)
        self.assertEqual("const int var1", r.gen_arg_as_c())
        self.assertEqual(      "int var1", r.gen_arg_as_c(asgn_value=True))
        self.assertEqual("const int var1", r.gen_arg_as_cxx())
        self.assertEqual(      "int var1", r.gen_arg_as_cxx(asgn_value=True))
        self.assertEqual(    "int * var1", r.gen_arg_as_cxx(asgn_value=True, force_ptr=True))

        r = declast.check_decl("int const var1")
        s = r.gen_decl()
        self.assertEqual("const int var1", s)

        r = declast.check_decl("int *var1 +dimension(:)")
        self.assertIsNone(r.get_subprogram())
        self.assertEqual(1, r.is_pointer())
        s = r.gen_decl()
        self.assertEqual("int * var1 +dimension(:)", s)
        self.assertEqual("int * var1", r.gen_arg_as_c())
        self.assertEqual("int var1", r.gen_arg_as_c(as_scalar=True))
        self.assertEqual("int * var1", r.gen_arg_as_cxx())
        self.assertEqual("integer(C_INT) :: var1(:)", r.gen_arg_as_fortran())
        self.assertEqual("integer(C_INT), pointer :: var1(:)",
                         r.gen_arg_as_fortran(attributes=['pointer']))
        self.assertEqual("integer(C_INT), pointer :: var1(:)",
                         r.gen_arg_as_fortran(is_pointer=True))
        self.assertEqual("integer(C_INT) :: var1(*)", r.bind_c())

        r = declast.check_decl("const int * var1")
        s = r.gen_decl()
        self.assertEqual("const int * var1", s)
        self.assertEqual("const int * var1", r.gen_arg_as_c())
        self.assertEqual("const int * var1", r.gen_arg_as_c(asgn_value=True))
        self.assertEqual("const int * var1", r.gen_arg_as_cxx())
        self.assertEqual("const int * var1", r.gen_arg_as_cxx(asgn_value=True))

        r = declast.check_decl("int * const var1")
        s = r.gen_decl()
        self.assertEqual("int * const var1", s)

        r = declast.check_decl("int **var1")
        s = r.gen_decl()
        self.assertEqual("int * * var1", s)

        r = declast.check_decl("const int * const * const var1")
        s = r.gen_decl()
        self.assertEqual("const int * const * const var1", s)

        r = declast.check_decl("long long var2")
        s = r.gen_decl()
        self.assertEqual("long long var2", s)

        # test attributes
        r = declast.check_decl("int m_ivar +readonly +name(ivar)")
        self.assertEqual(todict.to_dict(r),{
            "attrs": {
                "_typename": "int",
                "name": "ivar",
                "readonly": True,
            },
            "const": False,
            "declarator": {
                "name": "m_ivar",
                "pointer": [],
            },
            "specifier": [
                "int"
            ],
            "typemap_name": "int",
        })

    def test_type_string(self):
        """Test string declarations
        """
        typemap.initialize()

        r = declast.check_decl("char var1")
        s = r.gen_decl()
        self.assertEqual("char var1", s)

        r = declast.check_decl("char *var1")
        s = r.gen_decl()
        self.assertEqual("char * var1", s)
        s = r.gen_arg_as_fortran()
        self.assertEqual("character(len=*) :: var1", s)

        r = declast.check_decl("char *var1 +len(30)")
        s = r.gen_decl()
        self.assertEqual("char * var1 +len(30)", s)
        s = r.gen_arg_as_fortran(local=True)
        self.assertEqual("character(len=30) :: var1", s)

        r = declast.check_decl("char *var1 +allocatable")
        s = r.gen_decl()
        self.assertEqual("char * var1 +allocatable", s)
        s = r.gen_arg_as_fortran()
        self.assertEqual("character(len=:), allocatable :: var1", s)

        r = declast.check_decl("char *var1 +deref(allocatable)")
        s = r.gen_decl()
        self.assertEqual("char * var1 +deref(allocatable)", s)
        s = r.gen_arg_as_fortran()
        self.assertEqual("character(len=:), allocatable :: var1", s)

        r = declast.check_decl("char **var1")
        s = r.gen_decl()
        self.assertEqual("char * * var1", s)

        r = declast.check_decl("std::string var1")
        s = r.gen_decl()
        self.assertEqual("std::string var1", s)

        r = declast.check_decl("std::string *var1")
        s = r.gen_decl()
        self.assertEqual("std::string * var1", s)
        s = r.gen_arg_as_cxx()
        self.assertEqual("std::string * var1", s)
        s = r.gen_arg_as_c()
        self.assertEqual("char * var1", s)

        r = declast.check_decl("std::string &var1")
        s = r.gen_decl()
        self.assertEqual("std::string & var1", s)
        s = r.gen_arg_as_cxx()
        self.assertEqual("std::string & var1", s)
        s = r.gen_arg_as_cxx(as_ptr=True)
        self.assertEqual("std::string * var1", s)
        s = r.gen_arg_as_c()
        self.assertEqual("char * var1", s)

    def test_type_vector(self):
        """Test vector declarations
        """
        r = declast.check_decl("std::vector<int> var1")
        s = r.gen_decl()
        self.assertEqual("std::vector<int> var1", s)
        self.assertEqual(todict.to_dict(r),{
            "attrs": {
                "_typename": "std::vector",
                "template": "int"
            },
            "const": False,
            "declarator": {
                "name": "var1",
                "pointer": []
            },
            "specifier": ["std::vector"],
            'template_arguments': [
                {
                    'attrs': {'_typename': 'int'},
                    'const': False,
                    'specifier': ['int'],
                    'typemap_name': 'int'
                }
            ],
            "typemap_name": "std::vector",
        })

        r = declast.check_decl("std::vector<long long> var1")
        s = r.gen_decl()
        self.assertEqual("std::vector<long long> var1", s)
        self.assertEqual(todict.to_dict(r),{
            "attrs": {
                "_typename": "std::vector",
                "template": "long long"
            },
            "const": False,
            "declarator": {
                "name": "var1",
                "pointer": []
            },
            "specifier": ["std::vector"],
            'template_arguments': [
                {
                    'attrs': {'_typename': 'long_long'},
                    'const': False,
                    'specifier': ['long', 'long'],
                    'typemap_name': 'long_long'
                }
            ],
            "typemap_name": "std::vector",
        })

        r = declast.check_decl("std::vector<std::string> var1")
        s = r.gen_decl()
        self.assertEqual("std::vector<std::string> var1", s)
        self.assertEqual(todict.to_dict(r),{
            "attrs": {
                "_typename": "std::vector",
                "template": "std::string"
            },
            "const": False,
            "declarator": {
                "name": "var1",
                "pointer": []
            },
            "specifier": ["std::vector"],
            'template_arguments': [
                {
                    'attrs': {'_typename': 'std::string'},
                    'const': False,
                    'specifier': ['std::string'],
                    'typemap_name': 'std::string'
                }
            ],
            "typemap_name": "std::vector",
        })

    def test_template_argument_list(self):
        decl = "<int>"
        parser = declast.Parser(decl, None)
        r = parser.template_argument_list()
        self.assertEqual(todict.to_dict(r),
            [
                {
                    "attrs": {
                        "_typename": "int"
                    },
                    "const": False,
                    "specifier": [
                        "int"
                    ],
                    "typemap_name": "int",
                }
            ])

        # self.library creates a global namespace with std::string
        decl = "<std::string, int>"
        parser = declast.Parser(decl, self.library)
        r = parser.template_argument_list()
        self.assertEqual(todict.to_dict(r),
            [
                {
                    "attrs": {
                        "_typename": "std::string"
                    },
                    "const": False,
                    "specifier": [
                        "std::string"
                    ],
                    "typemap_name": "std::string",
                },
                {
                    "attrs": {
                        "_typename": "int"
                    },
                    "const": False,
                    "specifier": [
                        "int"
                    ],
                    "typemap_name": "int",
                }
            ])

    def test_declaration_specifier_error(self):
        with self.assertRaises(RuntimeError) as context:
            declast.check_decl("none var1")
        self.assertTrue("Expected TYPE_SPECIFIER, found ID 'none'" in str(context.exception))

        with self.assertRaises(RuntimeError) as context:
            declast.check_decl("std::int var1")
        self.assertTrue('Expected ID, found TYPE_SPECIFIER' in str(context.exception))

        with self.assertRaises(RuntimeError) as context:
            declast.check_decl("std::none var1")
        self.assertTrue("Symbol 'none' is not in namespace 'std'" in str(context.exception))

    def test_type_other(self):
        """Test size_t declarations
        """
        r = declast.check_decl("size_t var1()")
        s = r.gen_decl()
        self.assertEqual("size_t var1()", s)

        r = declast.check_decl("MPI_Comm get_comm()")
        s = r.gen_decl()
        self.assertEqual("MPI_Comm get_comm()", s)

    def test_type_int_func(self):
        """Test function declarations
        Test keywords of gen_decl.
        """
        r = declast.check_decl("int var1(int arg1) const")
        s = r.gen_decl()
        self.assertEqual("int var1(int arg1) const", s)

        s = r.gen_decl(params=None)
        self.assertEqual("int var1", s)

        s = r.gen_decl(name='newname', params=None)
        self.assertEqual("int newname", s)

        self.assertEqual('int', r.typename)
        self.assertFalse(r.is_pointer())
        self.assertEqual('function', r.get_subprogram())

        self.assertIsNotNone(r.find_arg_by_name('arg1'))
        self.assertIsNone(r.find_arg_by_name('argnone'))

    def test_type_function_pointer1(self):
        """Function pointer
        """
        r = declast.check_decl("int (*func)(int)")

        s = r.gen_decl()
        self.assertEqual("int ( * func)(int)", s)

        self.assertEqual("int", r.typename)
        self.assertEqual("func", r.name)
        self.assertFalse(r.is_pointer())
        self.assertFalse(r.is_reference())
        self.assertTrue(r.is_function_pointer())

        self.assertNotEqual(None, r.params)
        self.assertEqual(1, len(r.params))

        param0 = r.params[0]
        s = param0.gen_decl()
        self.assertEqual("int", s)
        self.assertEqual("int", param0.typename)
        self.assertEqual(None, param0.name)
        self.assertFalse(param0.is_pointer())
        self.assertFalse(param0.is_reference())
        self.assertFalse(param0.is_function_pointer())

        s = r.gen_decl()
        self.assertEqual("int ( * func)(int)", s)
        s = r.gen_arg_as_c()
        self.assertEqual("int ( * func)(int)", s)
        s = r.gen_arg_as_cxx()
        self.assertEqual("int ( * func)(int)", s)

        self.assertEqual(todict.to_dict(r),{
            "attrs": {
                "_typename": "int",
            },
            "const": False, 
            "declarator": {
                "func": {
                    "name": "func", 
                    "pointer": [
                        {
                            "const": False, 
                            "ptr": "*"
                        }
                    ]
                }, 
                "pointer": []
            }, 
            "func_const": False, 
            "params": [
                {
                    "attrs": {
                        "_typename": "int",
                    },
                    "const": False, 
                    "specifier": [
                        "int"
                    ],
                    "typemap_name": "int",
                }
            ], 
            "specifier": [
                "int"
            ],
            "typemap_name": "int",
        })
       
    def test_type_function_pointer2(self):
        """Function pointer
        """
        r = declast.check_decl("int *(*func)(int *arg)")

        s = r.gen_decl()
        self.assertEqual("int * ( * func)(int * arg)", s)

        self.assertEqual("int", r.typename)
        self.assertEqual("func", r.name)
        self.assertTrue(r.is_pointer())
        self.assertFalse(r.is_reference())
        self.assertTrue(r.is_function_pointer())

        self.assertNotEqual(None, r.params)
        self.assertEqual(1, len(r.params))

        param0 = r.params[0]
        s = param0.gen_decl()
        self.assertEqual("int * arg", s)
        self.assertEqual("int", param0.typename)
        self.assertEqual("arg", param0.name)
        self.assertTrue(param0.is_pointer())
        self.assertFalse(param0.is_reference())
        self.assertFalse(param0.is_function_pointer())

    # decl
    def test_decl01(self):
        """Simple declaration"""
        r = declast.check_decl("void foo")

        s = r.gen_decl()
        self.assertEqual("void foo", s)

        self.assertEqual(todict.to_dict(r),{
            "attrs": {
                "_typename": "void",
            },
            "const": False, 
            "declarator": {
                "name": "foo", 
                "pointer": []
            }, 
            "specifier": [
                "void"
            ],
            "typemap_name": "void",
        })
        self.assertEqual("foo", r.get_name())

    def test_decl02(self):
        """Simple declaration with attribute"""
        r = declast.check_decl("void foo +alias(junk)")

        s = r.gen_decl()
        self.assertEqual("void foo +alias(junk)", s)

        self.assertEqual(todict.to_dict(r),{
            "attrs": {
                "_typename": "void",
                "alias": "junk"
            }, 
            "const": False, 
            "declarator": {
                "name": "foo", 
                "pointer": []
            }, 
            "specifier": [
                "void"
            ], 
            "typemap_name": "void",
        })
        self.assertEqual("foo", r.get_name())

    def test_decl03(self):
        """Empty parameter list"""
        r = declast.check_decl("void foo()")

        s = r.gen_decl()
        self.assertEqual("void foo()", s)

        self.assertEqual(todict.to_dict(r),{
            "attrs": {
                "_typename": "void",
            },
            "const": False, 
            "declarator": {
                "name": "foo", 
                "pointer": []
            }, 
            "func_const": False, 
            "params": [], 
            "specifier": [
                "void"
            ], 
            "typemap_name": "void",
        })
        self.assertEqual("foo", r.get_name())
        self.assertEqual('void', r.typename)
        self.assertFalse(r.is_pointer())
        self.assertEqual('subroutine', r.get_subprogram())
        self.assertIsNone(r.find_arg_by_name('argnone'))

    def test_decl04(self):
        """const method"""
        r = declast.check_decl("void *foo() const")

        s = r.gen_decl()
        self.assertEqual("void * foo() const", s)

        self.assertEqual(todict.to_dict(r),{
            "attrs": {
                "_typename": "void",
            },
            "const": False, 
            "declarator": {
                "name": "foo", 
                "pointer": [ 
                    {
                        "const": False, 
                        "ptr": "*"
                    }
                ]
            }, 
            "func_const": True, 
            "params": [],
            "specifier": [
                "void"
            ],
            "typemap_name": "void",
        })
        self.assertEqual("foo", r.get_name())
        self.assertEqual('void', r.typename)
        self.assertTrue(r.is_pointer())
        self.assertEqual('function', r.get_subprogram())

    def test_decl05(self):
        """Single argument"""
        r = declast.check_decl("void foo(int arg1)")

        s = r.gen_decl()
        self.assertEqual("void foo(int arg1)", s)

        self.assertEqual(todict.to_dict(r),{
            "attrs": {
                "_typename": "void",
            },
            "const": False, 
            "declarator": {
                "name": "foo", 
                "pointer": []
            }, 
            "func_const": False, 
            "params": [
                {
                    "attrs": {
                        "_typename": "int",
                    },
                    "const": False, 
                    "declarator": {
                        "name": "arg1", 
                        "pointer": []
                    }, 
                    "specifier": [
                        "int"
                    ], 
                    "typemap_name": "int",
                }
            ], 
            "specifier": [
                "void"
            ], 
            "typemap_name": "void",
        })
        self.assertEqual("foo", r.get_name())

    def test_decl06(self):
        """multiple arguments"""
        r = declast.check_decl("void foo(int arg1, double arg2)")

        s = r.gen_decl()
        self.assertEqual("void foo(int arg1, double arg2)", s)

        self.assertEqual(todict.to_dict(r),{
            "attrs": {
                "_typename": "void",
            },
            "const": False, 
            "declarator": {
                "name": "foo", 
                "pointer": []
            }, 
            "func_const": False, 
            "params": [
                {
                    "attrs": {
                        "_typename": "int",
                    },
                    "const": False, 
                    "declarator": {
                        "name": "arg1", 
                        "pointer": []
                    }, 
                    "specifier": [
                        "int"
                    ],
                    "typemap_name": "int",
                }, 
                {
                    "attrs": {
                        "_typename": "double",
                    },
                    "const": False, 
                    "declarator": {
                        "name": "arg2", 
                        "pointer": []
                    }, 
                    "specifier": [
                        "double"
                    ], 
                    "typemap_name": "double",
                }
            ], 
            "specifier": [
                "void"
            ],
            "typemap_name": "void",
        })
        self.assertEqual("foo", r.get_name())

        self.assertIsNotNone(r.find_arg_by_name('arg1'))
        self.assertIsNotNone(r.find_arg_by_name('arg2'))
        self.assertIsNone(r.find_arg_by_name('argnone'))

    def test_decl07(self):
        """Return string"""
        r = declast.check_decl("const std::string& getName() const")

        s = r.gen_decl()
        self.assertEqual("const std::string & getName() const", s)
        self.assertFalse(r.is_pointer())
        self.assertTrue(r.is_reference())
        self.assertTrue(r.is_indirect())

        self.assertEqual(todict.to_dict(r),{
            "attrs": {
                "_typename": "std::string",
            },
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
            "func_const": True, 
            "params": [], 
            "specifier": [
                "std::string"
            ], 
            "typemap_name": "std::string",
        })
        self.assertEqual("getName", r.get_name())

    def test_decl08(self):
        """Test attributes.
        """
        r = declast.check_decl("const void foo("
                               "int arg1+in, double arg2+out)"
                               "+len=30 +attr2(True)" )

        s = r.gen_decl()
        self.assertEqual("const void foo("
                         "int arg1 +in, double arg2 +out)"
                         " +attr2(True)+len(30)", s)

        self.assertEqual(todict.to_dict(r),{
            "attrs": {
                "_typename": "void",
                "attr2": "True",
                "len": 30
            },
            "const": True, 
            "declarator": {
                "name": "foo", 
                "pointer": []
            }, 
            "func_const": False, 
            "params": [
                {
                    "attrs": {
                        "_typename": "int",
                        "in": True
                    }, 
                    "const": False, 
                    "declarator": {
                        "name": "arg1", 
                        "pointer": []
                    }, 
                    "specifier": [
                        "int"
                    ], 
                    "typemap_name": "int",
                }, 
                {
                    "attrs": {
                        "_typename": "double",
                        "out": True
                    }, 
                    "const": False, 
                    "declarator": {
                        "name": "arg2", 
                        "pointer": []
                    }, 
                    "specifier": [
                        "double"
                    ], 
                    "typemap_name": "double",
                }
            ], 
            "specifier": [
                "void"
            ],
            "typemap_name": "void",
        })
        self.assertEqual("foo", r.get_name())

    def test_decl09a(self):
        """Test constructor
        """
        r = declast.check_decl("Class1()",namespace=self.class1)

        s = r.gen_decl()
        self.assertEqual("Class1()", s)

        self.assertEqual(todict.to_dict(r),{
            "attrs": {
                "_constructor": True,
                "_name": "ctor",
                "_typename": "Class1",
            },
            "const": False,
            "func_const": False,
            "params": [],
            "specifier": [
                "Class1"
            ],
            "typemap_name": "Class1",
        })
        self.assertEqual('ctor', r.get_name())
        self.assertFalse(r.is_pointer())
        self.assertFalse(r.is_reference())
        # must provide the name since the ctor has no name
        self.assertEqual('Class1 ctor()', r.gen_arg_as_cxx())
        self.assertEqual('CC_class1 ctor', r.gen_arg_as_c(params=None))

    def test_decl09b(self):
        """Test constructor +name
        """
        r = declast.check_decl("Class1() +name(new)",namespace=self.class1)

        s = r.gen_decl()
        self.assertEqual("Class1() +name(new)", s)

        self.assertEqual(todict.to_dict(r),{
            "attrs": {
                "_constructor": True,
                "_name": "ctor",
                "_typename": "Class1",
                "name": "new",
            },
            "const": False,
            "func_const": False,
            "params": [],
            "specifier": [
                "Class1"
            ],
            "typemap_name": "Class1",
        })
        self.assertEqual('new', r.get_name())
        self.assertFalse(r.is_pointer())
        self.assertFalse(r.is_reference())
        self.assertFalse(r.is_indirect())
        self.assertEqual('Class1 new', r.gen_arg_as_cxx(params=None))
        self.assertEqual('CC_class1 new()', r.gen_arg_as_c())

    def test_decl09c(self):
        """Test destructor
        """
        r = declast.check_decl("~Class1()",namespace=self.class1)

        s = r.gen_decl()
        self.assertEqual("~Class1()", s)

        self.assertEqual(todict.to_dict(r),{
            "attrs": {
                "_destructor": True,
                "_name": "dtor",
                "_typename": "Class1",
            },
            "const": False,
            "func_const": False,
            "params": [],
            "specifier": [
                "Class1"
            ],
            "typemap_name": "Class1",
        })
        self.assertEqual('dtor', r.get_name())
        self.assertFalse(r.is_pointer())
        self.assertFalse(r.is_reference())
        self.assertFalse(r.is_indirect())
        self.assertEqual('Class1 dtor()', r.gen_arg_as_cxx())
        self.assertEqual('CC_class1 dtor()', r.gen_arg_as_c())

    def test_decl09d(self):
        """Return pointer to Class instance
        """
        r = declast.check_decl("Class1 * make()",namespace=self.class1)

        s = r.gen_decl()
        self.assertEqual("Class1 * make()", s)

        self.assertEqual(todict.to_dict(r),{
            "attrs": {
                "_typename": "Class1",
            },
            "const": False,
            "declarator": {
                "name": "make",
                "pointer": [
                    {
                        "const": False,
                        "ptr": "*"
                    }
            ]
            },
            "func_const": False,
            "params": [],
            "specifier": [
                "Class1"
            ],
            "typemap_name": "Class1",
        })
        self.assertEqual('make', r.get_name())

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

        self.assertEqual(todict.to_dict(r),{
            "attrs": {
                "_typename": "void",
            }, 
            "const": False, 
            "declarator": {
                "name": "name", 
                "pointer": []
            }, 
            "func_const": False, 
            "params": [
                {
                    "attrs": {
                        "_typename": "int",
                    }, 
                    "const": False, 
                    "declarator": {
                        "name": "arg1", 
                        "pointer": []
                    }, 
                    "init": 0,
                    "specifier": [
                        "int"
                    ],
                    "typemap_name": "int",
                }, 
                {
                    "attrs": {
                        "_typename": "double",
                    },
                    "const": False, 
                    "declarator": {
                        "name": "arg2", 
                        "pointer": []
                    }, 
                    "init": 0.0,
                    "specifier": [
                        "double"
                    ], 
                    "typemap_name": "double",
                }, 
                {
                    "attrs": {
                        "_typename": "std::string",
                    },
                    "const": False, 
                    "declarator": {
                        "name": "arg3", 
                        "pointer": []
                    }, 
                    "init": '"name"',
                    "specifier": [
                        "std::string"
                    ], 
                    "typemap_name": "std::string",
                }, 
                {
                    "attrs": {
                        "_typename": "bool",
                    },
                    "const": False, 
                    "declarator": {
                        "name": "arg4", 
                        "pointer": []
                    }, 
                    "init": "true",
                    "specifier": [
                        "bool"
                    ], 
                    "typemap_name": "bool",
                }
            ], 
            "specifier": [
                "void"
            ],
            "typemap_name": "void",
        })
        self.assertEqual("name", r.get_name())

    def test_decl11(self):
        """Test template_types
        """
        r = declast.check_decl("void decl11(ArgType arg)", template_types=['ArgType'])

        s = r.gen_decl()
        self.assertEqual("void decl11(ArgType arg)", s)

        self.assertEqual(todict.to_dict(r),{
            "attrs": {
                "_typename": "void",
            },
            "const": False, 
            "declarator": {
                "name": "decl11", 
                "pointer": []
            }, 
            "func_const": False, 
            "params": [
                {
                    "attrs": {
                        "_typename": "ArgType",
                    }, 
                    "const": False, 
                    "declarator": {
                        "name": "arg", 
                        "pointer": []
                    }, 
                    "specifier": [
                        "ArgType"
                    ], 
#XXX -                    "typemap_name": "ArgType",
                }
            ], 
            "specifier": [
                "void"
            ],
            "typemap_name": "void",
        })
        self.assertEqual("decl11", r.get_name())
                         
    def test_decl12(self):
        """Test templates
        Test std::string and string types.
        """
        r = declast.check_decl("void decl12(std::vector<std::string> arg1, string arg2)")

        s = r.gen_decl()
        self.assertEqual("void decl12(std::vector<std::string> arg1, string arg2)", s)

        self.assertEqual(todict.to_dict(r),{
            "attrs": {
                "_typename": "void",
            },
            "const": False, 
            "declarator": {
                "name": "decl12", 
                "pointer": []
            }, 
            "func_const": False, 
            "params": [
                {
                    "attrs": {
                        "_typename": "std::vector",
                        "template": "std::string",
                    }, 
                    "const": False, 
                    "declarator": {
                        "name": "arg1", 
                        "pointer": []
                    }, 
                    "specifier": ["std::vector"],
                    "template_arguments": [
                        {
                            'attrs': {'_typename': 'std::string'},
                            'const': False,
                            'specifier': ['std::string'],
                            'typemap_name': 'std::string'
                        },
                    ],
                    "typemap_name": "std::vector",
                }, 
                {
                    "attrs": {
                        "_typename": "std::string",
                    },
                    "const": False, 
                    "declarator": {
                        "name": "arg2", 
                        "pointer": []
                    }, 
                    "specifier": [
                        "string"
                    ], 
                    "typemap_name": "std::string",
                }
            ], 
            "specifier": [
                "void"
            ],
            "typemap_name": "void",
        })
        self.assertEqual("decl12", r.get_name())

    def test_decl13(self):
        """Test multi-specifier
        """
        r = declast.check_decl("void decl13("
                               "long int arg1,"
                               "long long arg2,"
                               "unsigned int)")

        self.assertEqual("long_int", r.params[0].typename)
        self.assertEqual("long_long", r.params[1].typename)
        self.assertEqual("unsigned_int", r.params[2].typename)

    def test_class_template(self):
        """Class templates"""
        r = declast.check_decl("template<typename T> class vector")

#        s = r.gen_decl()
#        self.assertEqual("template<typename T> vector", s)

        self.assertEqual(todict.to_dict(r),{
            "decl": {
                "name": "vector"
            },
            "parameters": [
                {
                    "name": "T"
                }
            ]
        })

        r = declast.check_decl("template<Key,T> class map")

#        s = r.gen_decl()
#        self.assertEqual("template<typename Key, typename T> map", s)

        self.assertEqual(todict.to_dict(r),{
            "decl": {
                "name": "map"
            },
            "parameters": [
                {
                    "name": "Key",
                },
                {
                    "name": "T",
                }
            ]
        })

    def test_as_arg(self):
        r = declast.check_decl("const std::string& getName() const")

        s = r.gen_decl()
        self.assertEqual("const std::string & getName() const", s)

        r.result_as_arg('output')
        s = r.gen_decl()
        self.assertEqual("void getName(std::string & output) const", s)

    def test_as_voidstar(self):
        # Change function return to an argument
        r = declast.check_decl("const std::string& getName() const")

        s = r.gen_decl()
        self.assertEqual("const std::string & getName() const", s)

        r.result_as_voidstar('void', 'output', const=r.const)
        s = r.gen_decl()
        self.assertEqual("void getName(const void * output) const", s)

        # Now without const
        r = declast.check_decl("std::string& getName() const")
        r.result_as_voidstar('void', 'output', const=r.const)
        s = r.gen_decl()
        self.assertEqual("void getName(void * output) const", s)
                         
    def test_thisarg01(self):
        """Create an argument for const this"""
        class_typemap = self.class1.typemap
        r = declast.create_this_arg('self', class_typemap, const=True)
        s = r.gen_decl()
        self.assertEqual("const Class1 * self", s)

    def test_thisarg02(self):
        """Create an argument for this"""
        class_typemap = self.class1.typemap
        r = declast.create_this_arg('self', class_typemap, const=False)
        s = r.gen_decl()
        self.assertEqual("Class1 * self", s)

    def test_copy01(self):
        """Test copy"""
        r = declast.check_decl("const std::string& Function4b("
                               "const std::string& arg1,"
                               "const std::string& arg2 )")
        self.assertTrue(r.is_reference())
        self.assertEqual(r.name, "Function4b")

        r2 = copy.deepcopy(r)

        r2.name = 'newname'
        self.assertEqual(r.name, 'Function4b')  # first is unchanged
        self.assertEqual(r2.name, 'newname')


class CheckExpr(unittest.TestCase):
    # No need for namespace

    def test_identifier1(self):
        r = declast.check_expr('id')
        self.assertEqual('id', todict.print_node(r))
        self.assertEqual(todict.to_dict(r),{
            'name' : 'id',
        })

    def test_identifier_no_args(self):
        r = declast.check_expr('id()')
        self.assertEqual('id()', todict.print_node(r))
        self.assertEqual(todict.to_dict(r),{
            'name' : 'id',
            'args': [],
        })

    def test_identifier_with_args(self):
        r = declast.check_expr('id(arg1)')
        self.assertEqual('id(arg1)', todict.print_node(r))
        self.assertEqual(todict.to_dict(r), {
            'name' : 'id',
            'args' : [ { 'name' : 'arg1' } ]
        })

    def test_constant(self):
        r = declast.check_expr('1 + 2.345')
        self.assertEqual('1+2.345', todict.print_node(r))
        self.assertEqual(todict.to_dict(r),{
            "left": {
                "constant": "1"
            }, 
            "op": "+", 
            "right": {
                "constant": "2.345"
            }
        })

    def test_binary(self):
        r = declast.check_expr('a + b * c')
        self.assertEqual('a+b*c', todict.print_node(r))
        self.assertEqual(todict.to_dict(r),{
            "left": {
                "name": "a"
            }, 
            "op": "+", 
            "right": {
                "left": {
                    "name": "b"
                }, 
                "op": "*", 
                "right": {
                    "name": "c"
                }
            }
        })

        r = declast.check_expr('(a + b) * c')
        self.assertEqual('(a+b)*c', todict.print_node(r))
        self.assertEqual(todict.to_dict(r),{
            "left": {
                "node": {
                    "left": {
                        "name": "a"
                    }, 
                    "op": "+", 
                    "right": {
                        "name": "b"
                    }
                }
            }, 
            "op": "*", 
            "right": {
                "name": "c"
            }
        })


    def test_others(self):
        e = 'size+2'
        r = declast.check_expr(e)
        self.assertEqual(e, todict.print_node(r))


class CheckNamespace(unittest.TestCase):
    def test_decl_namespace(self):
        """Parse a namespace"""
        r = declast.check_decl('namespace ns1')
        self.assertEqual('namespace ns1', todict.print_node(r))
        self.assertEqual(todict.to_dict(r),{
            'name': 'ns1',
        })


class CheckTypedef(unittest.TestCase):

    def XXXsetUp(self):
        library = ast.LibraryNode()

    def test_typedef1(self):
        r = declast.check_decl('typedef int TypeID;')
        self.assertEqual('typedef int TypeID', r.gen_decl())
        self.assertDictEqual(todict.to_dict(r),{
            "attrs": {
                "_typename": "int"
            }, 
            "const": False, 
            "declarator": {
                "name": "TypeID", 
                "pointer": []
            }, 
            "specifier": [
                "int"
            ], 
            "storage": [
                "typedef"
            ],
            "typemap_name": "int",
        })

    def test_typedef2(self):
        library = ast.LibraryNode()
        library.add_declaration('typedef int TD2;')
        self.assertIn('TD2', library.symbols)

        typedef = typemap.lookup_type('TD2')
        self.assertIsNotNone(typedef)
        self.assertEqual('TD2', typedef.name)
        self.assertEqual('TD2', typedef.cxx_type)
        self.assertEqual('int', typedef.typedef)

    def test_typedef_errors(self):
        with self.assertRaises(RuntimeError) as context:
            r = declast.check_decl('typedef none TypeID;')
        self.assertTrue("Expected TYPE_SPECIFIER, found ID 'none'"
                        in str(context.exception))

        library = ast.LibraryNode()
        with self.assertRaises(NotImplementedError) as context:
            library.add_declaration('typedef int * TD2;')
        self.assertTrue("Pointers not supported in typedef"
                        in str(context.exception))

        with self.assertRaises(NotImplementedError) as context:
            library.add_declaration('typedef int(*func)();')
        self.assertTrue("Function pointers not supported in typedef"
                        in str(context.exception))


class CheckEnum(unittest.TestCase):

    def XXXsetUp(self):
        library = ast.LibraryNode()

    def test_enum1(self):
        r = declast.check_decl('enum Color{RED=1,BLUE,WHITE}')
        self.assertEqual('enum Color { RED = 1, BLUE, WHITE };', todict.print_node(r))
        self.assertEqual(todict.to_dict(r),{
            'name': 'Color',
            'members': [
                {'name': 'RED', 'value': { 'constant': '1' }},
                {'name': 'BLUE'},
                {'name': 'WHITE'}
            ]
        })
                         
    def test_enum2(self):
        # enum trailing comma
        r = declast.check_decl('enum Color{RED=1,BLUE,WHITE,}')
        self.assertEqual('enum Color { RED = 1, BLUE, WHITE };', todict.print_node(r))


class CheckStruct(unittest.TestCase):
    def test_struct1(self):
        r = declast.check_decl('struct struct1 { int i; double d; };')
        self.assertEqual('struct struct1 { int i;double d; };', todict.print_node(r))
        self.assertEqual(todict.to_dict(r),{
            "members": [
                {
                    "attrs": {
                        "_typename": "int"
                    },
                    "const": False,
                    "declarator": {
                        "name": "i",
                        "pointer": []
                    },
                    "specifier": [
                        "int"
                    ],
                    "typemap_name": "int",
                },
                {
                    "attrs": {
                        "_typename": "double"
                    },
                    "const": False,
                    "declarator": {
                        "name": "d",
                        "pointer": []
                    },
                    "specifier": [
                        "double"
                    ],
                    "typemap_name": "double",
                }
            ],
            "name": "struct1"
        })


class CheckClass(unittest.TestCase):

    def XXXsetUp(self):
        library = ast.LibraryNode()

    def test_class1(self):
        r = declast.check_decl('class Class1')
        self.assertIsInstance(r, declast.CXXClass)
        self.assertEqual('class Class1;', todict.print_node(r))
        self.assertEqual(todict.to_dict(r),{
            'name': 'Class1',
        })
                         
    def test_class2(self):
        """Forward declare class in a library"""
        library = ast.LibraryNode()
        library.add_declaration('class Class1;')
        self.assertIn('Class1', library.symbols)
        self.assertIsInstance(library.symbols['Class1'], ast.TypedefNode)

        typedef = typemap.lookup_type('Class1')
        self.assertIsNotNone(typedef)
        self.assertEqual('Class1', typedef.name)
        self.assertEqual('Class1', typedef.cxx_type)
        self.assertEqual('shadow', typedef.base)
                         
    def test_class2_node(self):
        """Add a class with declarations to a library"""
        library = ast.LibraryNode()
        library.add_declaration('class Class1;',
                                declarations=[ dict(decl='void func1()')])
        self.assertIn('Class1', library.symbols)
        sym = library.symbols['Class1']
        self.assertIsInstance(sym, ast.ClassNode)
        self.assertIs(sym, library.classes[0])

    def test_class_in_namespace(self):
        """Forward declare a class in a namespace"""
        library = ast.LibraryNode()
        ns = library.add_namespace('ns')
        ns.add_declaration('class Class2;')
        self.assertIn('Class2', ns.symbols)
        self.assertIsInstance(ns.symbols['Class2'], ast.TypedefNode)

        typedef = typemap.lookup_type('ns::Class2')
        self.assertIsNotNone(typedef)
        self.assertEqual('ns::Class2', typedef.name)
        self.assertEqual('ns::Class2', typedef.cxx_type)


if __name__ == '__main__':
    unittest.main()
