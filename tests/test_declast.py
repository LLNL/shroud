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
import copy

class CheckParse(unittest.TestCase):
    maxDiff = None

    # types
    def test_type_int(self):
        """Test variable declarations
        Combinations of const and pointer.
        """
        r = declast.check_decl("int")
        s = r.gen_decl()
        self.assertEqual("int ", s)

        r = declast.check_decl("int var1")
        s = r.gen_decl()
        self.assertEqual("int var1", s)

        r = declast.check_decl("const int var1")
        s = r.gen_decl()
        self.assertEqual("const int var1", s)

        r = declast.check_decl("int const var1")
        s = r.gen_decl()
        self.assertEqual("const int var1", s)

        r = declast.check_decl("int *var1")
        s = r.gen_decl()
        self.assertEqual("int * var1", s)

        r = declast.check_decl("int * const var1")
        s = r.gen_decl()
        self.assertEqual("int * const var1", s)

        r = declast.check_decl("int **var1")
        s = r.gen_decl()
        self.assertEqual("int * * var1", s)

        r = declast.check_decl("const int * const * const var1")
        s = r.gen_decl()
        self.assertEqual("const int * const * const var1", s)

    def test_type_string(self):
        """Test string declarations
        """
        r = declast.check_decl("char var1")
        s = r.gen_decl()
        self.assertEqual("char var1", s)

        r = declast.check_decl("char *var1")
        s = r.gen_decl()
        self.assertEqual("char * var1", s)

        r = declast.check_decl("char **var1")
        s = r.gen_decl()
        self.assertEqual("char * * var1", s)

        r = declast.check_decl("std::string var1")
        s = r.gen_decl()
        self.assertEqual("std::string var1", s)

        r = declast.check_decl("std::string *var1")
        s = r.gen_decl()
        self.assertEqual("std::string * var1", s)

        r = declast.check_decl("std::string &var1")
        s = r.gen_decl()
        self.assertEqual("std::string & var1", s)

    # decl
    def test_decl01(self):
        """Simple declaration"""
        r = declast.check_decl("void foo")

        s = r.gen_decl()
        self.assertEqual("void foo", s)

        self.assertEqual(r._to_dict(),{
            "attrs": {}, 
            "const": False, 
            "declarator": {
                "name": "foo", 
                "pointer": []
            }, 
            "specifier": [
                "void"
            ], 
        })
        self.assertEqual("foo", r.get_name())

    def test_decl02(self):
        """Simple declaration with attribute"""
        r = declast.check_decl("void foo +alias(junk)")

        s = r.gen_decl()
        self.assertEqual("void foo +alias(junk)", s)

        self.assertEqual(r._to_dict(),{
            "attrs": {
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
        })
        self.assertEqual("foo", r.get_name())

    def test_decl03(self):
        """Empty parameter list"""
        r = declast.check_decl("void foo()")

        s = r.gen_decl()
        self.assertEqual("void foo()", s)

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
        })
        self.assertEqual("foo", r.get_name())

    def test_decl04(self):
        """const method"""
        r = declast.check_decl("void foo() const")

        s = r.gen_decl()
        self.assertEqual("void foo() const", s)

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
        })
        self.assertEqual("foo", r.get_name())

    def test_decl05(self):
        """Single argument"""
        r = declast.check_decl("void foo(int arg1)")

        s = r.gen_decl()
        self.assertEqual("void foo(int arg1)", s)

        self.assertEqual(r._to_dict(),{
            "args": [
                {
                    "attrs": {}, 
                    "const": False, 
                    "declarator": {
                        "name": "arg1", 
                        "pointer": []
                    }, 
                    "specifier": [
                        "int"
                    ], 
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
        })
        self.assertEqual("foo", r.get_name())

    def test_decl06(self):
        """multiple arguments"""
        r = declast.check_decl("void foo(int arg1, double arg2)")

        s = r.gen_decl()
        self.assertEqual("void foo(int arg1, double arg2)", s)

        self.assertEqual(r._to_dict(),{
            "args": [
                {
                    "attrs": {}, 
                    "const": False, 
                    "declarator": {
                        "name": "arg1", 
                        "pointer": []
                    }, 
                    "specifier": [
                        "int"
                    ], 
                }, 
                {
                    "attrs": {}, 
                    "const": False, 
                    "declarator": {
                        "name": "arg2", 
                        "pointer": []
                    }, 
                    "specifier": [
                        "double"
                    ], 
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
        })
        self.assertEqual("foo", r.get_name())

    def test_decl07(self):
        """Return string"""
        r = declast.check_decl("const std::string& getName() const")

        s = r.gen_decl()
        self.assertEqual("const std::string & getName() const", s)

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
        })
        self.assertEqual("getName", r.get_name())

    def test_decl08(self):
        """Test attributes.
        """
        r = declast.check_decl("const void foo+attr1(30)+len=30("
                               "int arg1+in, double arg2+out)"
                               "+attr2(True)" )

        s = r.gen_decl()
        self.assertEqual("const void foo +attr1(30)+len(30)("
                         "int arg1 +in, double arg2 +out)"
                         " +attr2(True)", s)

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
                    "specifier": [
                        "int"
                    ], 
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
                    "specifier": [
                        "double"
                    ], 
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
        })
        self.assertEqual("foo", r.get_name())

    def test_decl09(self):
        """Test constructor
        The type and varialbe have the same name.
        """
        r = declast.check_decl("Class1 *Class1()  +constructor",current_class='Class1')

        s = r.gen_decl()
        self.assertEqual("Class1 * Class1() +constructor", s)

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

        self.assertEqual(r._to_dict(),{
            "args": [
                {
                    "attrs": {}, 
                    "const": False, 
                    "declarator": {
                        "name": "arg1", 
                        "pointer": []
                    }, 
                    "init": 0,
                    "specifier": [
                        "int"
                    ], 
                }, 
                {
                    "attrs": {}, 
                    "const": False, 
                    "declarator": {
                        "name": "arg2", 
                        "pointer": []
                    }, 
                    "init": 0.0,
                    "specifier": [
                        "double"
                    ], 
                }, 
                {
                    "attrs": {}, 
                    "const": False, 
                    "declarator": {
                        "name": "arg3", 
                        "pointer": []
                    }, 
                    "init": '"name"',
                    "specifier": [
                        "std::string"
                    ], 
                }, 
                {
                    "attrs": {}, 
                    "const": False, 
                    "declarator": {
                        "name": "arg4", 
                        "pointer": []
                    }, 
                    "init": "true",
                    "specifier": [
                        "bool"
                    ], 
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
        })
        self.assertEqual("name", r.get_name())

    def test_decl11(self):
        """Test template_types
        """
        r = declast.check_decl("void decl11(ArgType arg)", template_types=['ArgType'])

        s = r.gen_decl()
        self.assertEqual("void decl11(ArgType arg)", s)

        self.assertEqual(r._to_dict(),{
            "args": [
                {
                    "attrs": {}, 
                    "const": False, 
                    "declarator": {
                        "name": "arg", 
                        "pointer": []
                    }, 
                    "specifier": [
                        "ArgType"
                    ], 
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
        })
        self.assertEqual("decl11", r.get_name())
                         
    def test_decl12(self):
        """Test templates
        Test std::string and string types.
        """
        r = declast.check_decl("void decl12(std::vector<std::string> arg1, string arg2)")

        s = r.gen_decl()
        self.assertEqual("void decl12(std::vector<std::string> arg1, string arg2)", s)

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
                    "specifier": [
                        "std::vector"
                    ], 
                }, 
                {
                    "attrs": {}, 
                    "const": False, 
                    "declarator": {
                        "name": "arg2", 
                        "pointer": []
                    }, 
                    "specifier": [
                        "string"
                    ], 
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

    def test_decl14(self):
        """Function pointer
        """
        r = declast.check_decl("int CallBack1(  int (*func)(int) )")

        s = r.gen_decl()
        self.assertEqual("int CallBack1(int (* func)(int ))", s)

        self.assertEqual(r._to_dict(),{
            "args": [
                {
                    "args": [
                        {
                            "attrs": {}, 
                            "const": False, 
                            "declarator": {
                                "pointer": []
                            }, 
                            "specifier": [
                                "int"
                            ], 
                        }
                    ], 
                    "attrs": {}, 
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
                    "fattrs": {}, 
                    "func_const": False, 
                    "specifier": [
                        "int"
                    ], 
                }
            ], 
            "attrs": {}, 
            "const": False, 
            "declarator": {
                "name": "CallBack1", 
                "pointer": []
            }, 
            "fattrs": {}, 
            "func_const": False, 
            "specifier": [
                "int"
            ], 
        })

    def test_asarg(self):
        r = declast.check_decl("const std::string& getName() const")

        s = r.gen_decl()
        self.assertEqual("const std::string & getName() const", s)

        r.result_as_arg('output')
        s = r.gen_decl()
        self.assertEqual("void getName(std::string & output) const", s)
                         
    def test_thisarg01(self):
        """Create an argument for const this"""
        r = declast.create_this_arg('self', 'Class1', const=True)
        s = r.gen_decl()
        self.assertEqual("const Class1 * self", s)

    def test_thisarg02(self):
        """Create an argument for this"""
        r = declast.create_this_arg('self', 'Class1', const=False)
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
        self.assertEqual(r.name, 'Function4b')
        self.assertEqual(r2.name, 'newname')

        r2.set_indirection()
        self.assertTrue(r.is_reference())    # first is unchanged
        self.assertFalse(r2.is_reference())

                         
if __name__ == '__main__':
    unittest.main()
