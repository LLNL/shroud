# Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and
# other Shroud Project Developers.
# See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
########################################################################
"""
Parse C++ declarations.
"""
from __future__ import print_function

from shroud import declast
from shroud import todict

import unittest
import copy

# Useful to format reference output of to_dict
#import pprint
#pp = pprint.PrettyPrinter(indent=4)
#        print(pp.pprint(todict.to_dict(r)))

class CheckParse(unittest.TestCase):
    maxDiff = None

    # types
    def test_type_int(self):
        """Test variable declarations
        Combinations of const and pointer.
        """
        symtab = declast.SymbolTable()

        r = declast.check_decl("int", symtab)
        self.assertIsNone(r.get_subprogram())
        self.assertEqual(0, r.is_pointer())
        s = r.gen_decl()
        self.assertEqual("int", s)
        self.assertEqual("scalar", r.get_indirect_stmt())
        self.assertEqual(None, r.get_array_size())

        r = declast.check_decl("int var1", symtab)
        self.assertIsNone(r.get_subprogram())
        self.assertEqual(0, r.is_pointer())
        s = r.gen_decl()
        self.assertEqual("int var1", s)
        s = r.bind_c()
        self.assertEqual("integer(C_INT) :: var1", s)
        s = r.bind_c(intent="out")
        self.assertEqual("integer(C_INT), intent(OUT) :: var1", s)
        s = r.gen_arg_as_fortran()
        self.assertEqual("integer(C_INT) :: var1", s)

        r = declast.check_decl("const int var1", symtab)
        self.assertIsNone(r.get_subprogram())
        self.assertEqual(0, r.is_pointer())
        s = r.gen_decl()
        self.assertEqual("const int var1", s)
        self.assertEqual("const int var1", r.gen_arg_as_c())
        self.assertEqual("int var1", r.gen_arg_as_c(asgn_value=True))
        self.assertEqual("const int var1", r.gen_arg_as_cxx())
        self.assertEqual("int var1", r.gen_arg_as_cxx(asgn_value=True))
        self.assertEqual(
            "int * var1", r.gen_arg_as_cxx(asgn_value=True, force_ptr=True)
        )
        self.assertEqual("int", r.as_cast())
        self.assertEqual(
            todict.to_dict(r), {
                'const': True,
                'declarator': {
                    'name': 'var1',
                    'typemap_name': 'int',
                },
                'specifier': ['int'],
                'typemap_name': 'int',
            })
        self.assertEqual("scalar", r.get_indirect_stmt())

        r = declast.check_decl("int const var1", symtab)
        s = r.gen_decl()
        self.assertEqual("const int var1", s)
        self.assertEqual(
            todict.to_dict(r), {
                'const': True,
                'declarator': {
                    'name': 'var1',
                    'typemap_name': 'int',
                },
                'specifier': ['int'],
                'typemap_name': 'int'})
        self.assertEqual("scalar", r.get_indirect_stmt())
        r = declast.check_decl("int *var1 +dimension(:)", symtab)
        self.assertIsNone(r.get_subprogram())
        self.assertEqual(1, r.is_pointer())
        s = r.gen_decl()
        self.assertEqual("int * var1 +dimension(:)", s)
        self.assertEqual("int * var1", r.gen_arg_as_c())
        self.assertEqual("int var1", r.gen_arg_as_c(as_scalar=True))
        self.assertEqual("int * var1", r.gen_arg_as_cxx())
        self.assertEqual("integer(C_INT) :: var1(:)", r.gen_arg_as_fortran())
        self.assertEqual("integer(C_INT) :: var1(*)", r.bind_c())

        r = declast.check_decl("const int * var1", symtab)
        s = r.gen_decl()
        self.assertEqual("const int * var1", s)
        self.assertEqual("const int * var1",
                         r.gen_arg_as_c())
        self.assertEqual("const int * var1",
                         r.gen_arg_as_c(asgn_value=True))
        self.assertEqual("const int * var1",
                         r.gen_arg_as_cxx())
        self.assertEqual("const int * var1",
                         r.gen_arg_as_cxx(asgn_value=True))
        self.assertEqual("int *", r.as_cast())

        r = declast.check_decl("int * const var1", symtab)
        s = r.gen_decl()
        self.assertEqual("int * const var1", s)
        self.assertEqual(
            todict.to_dict(r), {
                'declarator': {
                    'name': 'var1',
                    'pointer': [{'const': True, 'ptr': '*'}],
                    'typemap_name': 'int',
                },
                'specifier': ['int'],
                'typemap_name': 'int'})
        self.assertEqual("*", r.get_indirect_stmt())
        self.assertEqual(None, r.get_array_size())

        r = declast.check_decl("int **var1", symtab)
        s = r.gen_decl()
        self.assertEqual("int * * var1", s)
        self.assertEqual(
            todict.to_dict(r), {
                'declarator': {
                    'name': 'var1',
                    'pointer': [{'ptr': '*'}, {'ptr': '*'}],
                    'typemap_name': 'int'
                },
                'specifier': ['int'],
                'typemap_name': 'int'
            })
        self.assertEqual("**", r.get_indirect_stmt())
        self.assertEqual(None, r.get_array_size())

        r = declast.check_decl("int &*var1", symtab)
        s = r.gen_decl()
        self.assertEqual("int & * var1", s)
        self.assertEqual(
            todict.to_dict(r), {
                'declarator': {
                    'name': 'var1',
                    'pointer': [{   'ptr': '&'}, {   'ptr': '*'}],
                    'typemap_name': 'int',
                },
                'specifier': ['int'],
                'typemap_name': 'int',
            })
        self.assertEqual("&*", r.get_indirect_stmt())
        self.assertEqual("int **", r.as_cast())

        r = declast.check_decl("const int * const * const var1", symtab)
        s = r.gen_decl()
        self.assertEqual("const int * const * const var1", s)
        self.assertEqual(
            todict.to_dict(r), {
                'const': True,
                'declarator': {
                    'name': 'var1',
                    'pointer': [
                        {'const': True, 'ptr': '*'},
                        {'const': True, 'ptr': '*'}
                    ],
                    'typemap_name': 'int',
                },
                'specifier': ['int'],
                'typemap_name': 'int',
            })
        self.assertEqual("**", r.get_indirect_stmt())
        self.assertEqual("int **", r.as_cast())

        r = declast.check_decl("long long var2", symtab)
        s = r.gen_decl()
        self.assertEqual("long long var2", s)

        # test attributes
        r = declast.check_decl("int m_ivar +readonly +name(ivar)", symtab)
        self.assertEqual(
            todict.to_dict(r),
            {
                "declarator": {
                    "name": "m_ivar",
                    "attrs": {"name": "ivar", "readonly": True},
                    "typemap_name": "int",
                },
                "specifier": ["int"],
                "typemap_name": "int",
            },
        )

    def test_type_int_array(self):
        symtab = declast.SymbolTable()

        r = declast.check_decl("int var1[20]", symtab)
        self.assertEqual("int var1[20]", str(r))
        s = r.gen_decl()
        self.assertEqual("int var1[20]", s)
        self.assertEqual(
            {
                'declarator': {
                    'name': 'var1',
                    'array': [{ 'constant': '20'}],
                    'typemap_name': 'int',
                },
                'specifier': ['int'],
                'typemap_name': 'int',
            },
            todict.to_dict(r)
        )
        self.assertEqual("int *", r.as_cast())
        self.assertEqual(
            "int var1[20]",
            r.gen_arg_as_c())
        self.assertEqual(
            "integer(C_INT) :: var1(20)",
            r.gen_arg_as_fortran())
        
        r = declast.check_decl("int var2[20][10]", symtab)
        self.assertEqual("int var2[20][10]", str(r))
        s = r.gen_decl()
        self.assertEqual("int var2[20][10]", s)
        self.assertEqual(
            {'declarator': {
                'name': 'var2',
                'array': [
                    { 'constant': '20'},
                    { 'constant': '10'},
                ],
                'typemap_name': 'int',
            },
             'specifier': ['int'],
             'typemap_name': 'int'},
            todict.to_dict(r)
        )
        self.assertEqual("int *", r.as_cast())
        self.assertEqual(
            "int var2[20][10]",
            r.gen_arg_as_c())
        self.assertEqual(
            "integer(C_INT) :: var2(10,20)",
            r.gen_arg_as_fortran())
        
        r = declast.check_decl("int var3[DEFINE + 3]", symtab)
        self.assertEqual("int var3[DEFINE+3]", str(r))
        s = r.gen_decl()
        self.assertEqual("int var3[DEFINE+3]", s)
        self.assertEqual(
            {
             'declarator': {
                 'name': 'var3',
                'array': [
                    {'left': {'name': 'DEFINE'},
                     'op': '+',
                     'right': {'constant': '3'}}],
                 'typemap_name': 'int',
             },
             'specifier': ['int'],
             'typemap_name': 'int'},
            todict.to_dict(r)
        )
        self.assertEqual("int *", r.as_cast())
        self.assertEqual(
            "int var3[DEFINE+3]",
            r.gen_arg_as_c())
        self.assertEqual(
            "integer(C_INT) :: var3(DEFINE+3)",
            r.gen_arg_as_fortran())
       
    def test_type_string(self):
        """Test string declarations
        """
        symtab = declast.SymbolTable()
        symtab.create_std_namespace()
        symtab.using_directive("std")

        r = declast.check_decl("char var1", symtab)
        s = r.gen_decl()
        self.assertEqual("char var1", s)
        self.assertEqual("char", r.as_cast())

        r = declast.check_decl("char *var1", symtab)
        self.assertEqual("char *", r.as_cast())
        s = r.gen_decl()
        self.assertEqual("char * var1", s)
        s = r.gen_arg_as_fortran()
        self.assertEqual("character(len=*) :: var1", s)

        r = declast.check_decl("char *var1 +len(30)", symtab)
        s = r.gen_decl()
        self.assertEqual("char * var1 +len(30)", s)
        s = r.gen_arg_as_fortran(local=True)
        self.assertEqual("character(len=30) :: var1", s)

        r = declast.check_decl("char *var1 +allocatable", symtab)
        s = r.gen_decl()
        self.assertEqual("char * var1 +allocatable", s)
        s = r.gen_arg_as_fortran()
        self.assertEqual("character(len=:), allocatable :: var1", s)

        r = declast.check_decl("char *var1 +deref(allocatable)", symtab)
        s = r.gen_decl()
        self.assertEqual("char * var1 +deref(allocatable)", s)
        s = r.gen_arg_as_fortran()
        self.assertEqual("character(len=:), allocatable :: var1", s)

        r = declast.check_decl("char **var1", symtab)
        self.assertEqual(2, r.is_indirect())
        self.assertEqual(2, r.is_array())
        self.assertEqual('**', r.get_indirect_stmt())
        self.assertEqual("char **", r.as_cast())
        s = r.gen_decl()
        self.assertEqual("char * * var1", s)

        r = declast.check_decl("std::string var1", symtab)
        s = r.gen_decl()
        self.assertEqual("std::string var1", s)

        r = declast.check_decl("std::string *var1", symtab)
        s = r.gen_decl()
        self.assertEqual("std::string * var1", s)
        s = r.gen_arg_as_cxx()
        self.assertEqual("std::string * var1", s)
        s = r.gen_arg_as_c()
        self.assertEqual("char * var1", s)
        self.assertEqual("char *", r.as_cast())

        r = declast.check_decl("std::string &var1", symtab)
        s = r.gen_decl()
        self.assertEqual("std::string & var1", s)
        s = r.gen_arg_as_cxx()
        self.assertEqual("std::string & var1", s)
        s = r.gen_arg_as_cxx(as_ptr=True)
        self.assertEqual("std::string * var1", s)
        s = r.gen_arg_as_c()
        self.assertEqual("char * var1", s)

    def test_type_char_array(self):
        # convert first dimension to Fortran CHARACTER(LEN=)
        symtab = declast.SymbolTable()

        r = declast.check_decl("char var1[20]", symtab)
        self.assertEqual("char var1[20]", str(r))
        self.assertEqual(0, r.is_indirect())
        self.assertEqual(1, r.is_array())
        self.assertEqual("char *", r.as_cast())
        self.assertEqual('[]', r.get_indirect_stmt())
        self.assertEqual("20", r.get_array_size())
        s = r.gen_decl()
        self.assertEqual("char var1[20]", s)
        self.assertEqual(
            {'declarator': {
                'name': 'var1',
                'array': [{ 'constant': '20'}],
                'typemap_name': 'char',
            },
             'specifier': ['char'],
             'typemap_name': 'char'},
            todict.to_dict(r)
        )
        self.assertEqual(
            "char var1[20]",
            r.gen_arg_as_c())
        self.assertEqual(
            "character(kind=C_CHAR) :: var1(20)",
            r.gen_arg_as_fortran())
        
        r = declast.check_decl("char var2[20][10][5]", symtab)
        self.assertEqual(0, r.is_indirect())
        self.assertEqual(1, r.is_array())
        self.assertEqual("char *", r.as_cast())
        self.assertEqual('[]', r.get_indirect_stmt())
        self.assertEqual("(20)*(10)*(5)", r.get_array_size())
        self.assertEqual("char var2[20][10][5]", str(r))
        s = r.gen_decl()
        self.assertEqual("char var2[20][10][5]", s)
        self.assertEqual(
            {'declarator': {
                'name': 'var2',
                'array': [
                    { 'constant': '20'},
                    { 'constant': '10'},
                    { 'constant': '5'},
                ],
                'typemap_name': 'char',
            },
             'specifier': ['char'],
             'typemap_name': 'char'},
            todict.to_dict(r)
        )
        self.assertEqual(
            "char var2[20][10][5]",
            r.gen_arg_as_c())
        self.assertEqual(
            "character(kind=C_CHAR) :: var2(5,10,20)",
            r.gen_arg_as_fortran())
        
        r = declast.check_decl("char var3[DEFINE + 3]", symtab)
        self.assertEqual(0, r.is_indirect())
        self.assertEqual(1, r.is_array())
        self.assertEqual("char *", r.as_cast())
        self.assertEqual('[]', r.get_indirect_stmt())
        self.assertEqual("DEFINE+3", r.get_array_size())
        self.assertEqual("char var3[DEFINE+3]", str(r))
        s = r.gen_decl()
        self.assertEqual("char var3[DEFINE+3]", s)
        self.assertEqual(
            {
             'declarator': {
                 'name': 'var3',
                 'array': [
                     {'left': {'name': 'DEFINE'},
                      'op': '+',
                      'right': {'constant': '3'}}],
                 'typemap_name': 'char',
             },
            'specifier': ['char'],
             'typemap_name': 'char'},
            todict.to_dict(r)
        )
        self.assertEqual(
            "char var3[DEFINE+3]",
            r.gen_arg_as_c())
        self.assertEqual(
            "character(kind=C_CHAR) :: var3(DEFINE+3)",
            r.gen_arg_as_fortran())
    
        r = declast.check_decl("char *var4[44]", symtab)
        self.assertEqual("char * var4[44]", str(r))
        self.assertEqual(1, r.is_indirect())
        self.assertEqual(2, r.is_array())
        self.assertEqual("char **", r.as_cast())
        self.assertEqual('*[]', r.get_indirect_stmt())
        self.assertEqual("44", r.get_array_size())
        s = r.gen_decl()
        self.assertEqual("char * var4[44]", s)
        self.assertEqual(
            {
             'declarator': {
                 'name': 'var4',
                 'array': [{'constant': '44'}],
                 'pointer': [{'ptr': '*'}],
                 'typemap_name': 'char',
             },
             'specifier': ['char'],
             'typemap_name': 'char'},
            todict.to_dict(r)
        )
        self.assertEqual(
            "char * var4[44]",
            r.gen_arg_as_c())
        self.assertEqual(  # XXX - fixme
            "character(kind=C_CHAR) :: var4(44)",
            r.gen_arg_as_fortran())
    
    def test_type_vector(self):
        """Test vector declarations
        """
        symtab = declast.SymbolTable()
        symtab.create_std_namespace()
        symtab.using_directive("std")

        r = declast.check_decl("std::vector<int> var1", symtab)
        s = r.gen_decl()
        self.assertEqual("std::vector<int> var1", s)
        self.assertEqual(
            todict.to_dict(r),
            {
                "declarator": {
                    "name": "var1",
                    "typemap_name": "std::vector",
                },
                "specifier": ["std::vector"],
                "template_arguments": [
                    {"specifier": ["int"],
                     "typemap_name": "int",
                     "declarator": {
                         "typemap_name": "int",
                     },
                    }
                ],
                "typemap_name": "std::vector",
            },
        )
        # C
        s = r.gen_arg_as_c()
        self.assertEqual("int var1", s)
        s = r.gen_arg_as_c(force_ptr=True)
        self.assertEqual("int * var1", s)
        # CXX
        s = r.gen_arg_as_cxx()
        self.assertEqual("int var1", s)
        s = r.gen_arg_as_cxx(force_ptr=True)
        self.assertEqual("int * var1", s)

        s = r.gen_arg_as_cxx(force_ptr=True, with_template_args=True)
        self.assertEqual("std::vector<int> * var1", s)

        r = declast.check_decl("std::vector<long long> var1", symtab)
        s = r.gen_decl()
        self.assertEqual("std::vector<long long> var1", s)
        self.assertEqual(
            todict.to_dict(r),
            {
                "declarator": {
                    "name": "var1",
                    "typemap_name": "std::vector",
                },
                "specifier": ["std::vector"],
                "template_arguments": [
                    {
                        "specifier": ["long", "long"],
                        "typemap_name": "long_long",
                        "declarator": {
                            "typemap_name": "long_long",
                        },
                    }
                ],
                "typemap_name": "std::vector",
            },
        )

        r = declast.check_decl("std::vector<std::string> var1", symtab)
        s = r.gen_decl()
        self.assertEqual("std::vector<std::string> var1", s)
        self.assertEqual(
            todict.to_dict(r),
            {
                "declarator": {
                    "name": "var1",
                    "typemap_name": "std::vector",
                },
                "specifier": ["std::vector"],
                "template_arguments": [
                    {
                        "specifier": ["std::string"],
                        "typemap_name": "std::string",
                        "declarator": {
                            "typemap_name": "std::string",
                        },
                    }
                ],
                "typemap_name": "std::vector",
            },
        )

    def test_type_vector_ptr(self):
        """Test vector declaration with pointer
        """
        symtab = declast.SymbolTable()
        symtab.create_std_namespace()
        symtab.using_directive("std")

        r = declast.check_decl("std::vector<int *> var1", symtab)
        s = r.gen_decl()
        self.assertEqual("std::vector<int * > var1", s)
        self.assertEqual(
            todict.to_dict(r),
            {
                'declarator': {
                    'name': 'var1',
                    'typemap_name': 'std::vector'
                },
                'specifier': ['std::vector'],
                'template_arguments': [
                    {
                        'declarator': {
                            'pointer': [{'ptr': '*'}],
                            'typemap_name': 'int',
                        },
                        'specifier': ['int'],
                        'typemap_name': 'int',
                    }
                ],
                'typemap_name': 'std::vector'
            }
        )
        # C
        s = r.gen_arg_as_c()
        self.assertEqual("int var1", s)
        s = r.gen_arg_as_c(force_ptr=True)
        self.assertEqual("int * var1", s)
        # CXX
        s = r.gen_arg_as_cxx()
        self.assertEqual("int var1", s)
        s = r.gen_arg_as_cxx(force_ptr=True)
        self.assertEqual("int * var1", s)

        s = r.gen_arg_as_cxx(force_ptr=True, with_template_args=True)
        self.assertEqual("std::vector<int * > * var1", s)

    def test_template_argument_list(self):
        symtab = declast.SymbolTable()
        symtab.create_std_namespace()

        decl = "<int>"
        parser = declast.Parser(decl, symtab)
        r = parser.template_argument_list()
        self.assertEqual(
            todict.to_dict(r), [{"specifier": ["int"], "typemap_name": "int"}]
        )

        # self.library creates a global namespace with std::string
        decl = "<std::string, int>"
        parser = declast.Parser(decl, symtab)
        r = parser.template_argument_list()
        self.assertEqual(
            todict.to_dict(r),
            [
                {"specifier": ["std::string"], "typemap_name": "std::string"},
                {"specifier": ["int"], "typemap_name": "int"},
            ],
        )

    def test_declaration_specifier_error(self):
        symtab = declast.SymbolTable()
        symtab.create_std_namespace()
        symtab.using_directive("std")

        with self.assertRaises(RuntimeError) as context:
            declast.check_decl("none var1", symtab)
        self.assertTrue(
            "Expected TYPE_SPECIFIER, found ID 'none'" in str(context.exception)
        )

        with self.assertRaises(RuntimeError) as context:
            declast.check_decl("std::int var1", symtab)
        self.assertTrue(
            "Expected ID, found TYPE_SPECIFIER" in str(context.exception)
        )

        with self.assertRaises(RuntimeError) as context:
            declast.check_decl("std::none var1", symtab)
        self.assertTrue(
            "Symbol 'none' is not in namespace 'std'" in str(context.exception)
        )

    def test_type_other(self):
        """Test size_t declarations
        """
        symtab = declast.SymbolTable()
        symtab.create_std_names()  # size_t et al.
        
        r = declast.check_decl("size_t var1()", symtab)
        s = r.gen_decl()
        self.assertEqual("size_t var1(void)", s)

        r = declast.check_decl("MPI_Comm get_comm()", symtab)
        s = r.gen_decl()
        self.assertEqual("MPI_Comm get_comm(void)", s)

    def test_type_int_func(self):
        """Test function declarations
        Test keywords of gen_decl.
        """
        symtab = declast.SymbolTable()

        r = declast.check_decl("int var1(int arg1) const", symtab)
        s = r.gen_decl()
        self.assertEqual("int var1(int arg1) const", s)

        s = r.gen_decl(params=None)
        self.assertEqual("int var1", s)

        s = r.gen_decl(name="newname", params=None)
        self.assertEqual("int newname", s)

        self.assertEqual("int", r.typemap.name)
        self.assertFalse(r.is_pointer())
        self.assertEqual("function", r.get_subprogram())

        self.assertIsNotNone(r.find_arg_by_name("arg1"))
        self.assertIsNone(r.find_arg_by_name("argnone"))

    def test_type_function_pointer1(self):
        """Function pointer
        """
        symtab = declast.SymbolTable()

        r = declast.check_decl("int (*func)(int)", symtab)

        s = r.gen_decl()
        self.assertEqual("int ( * func)(int)", s)

        self.assertEqual("int", r.typemap.name)
        self.assertEqual("func", r.name)
        self.assertFalse(r.is_pointer())
        self.assertFalse(r.is_reference())
        self.assertTrue(r.is_function_pointer())

        self.assertNotEqual(None, r.params)
        self.assertEqual(1, len(r.params))

        param0 = r.params[0]
        s = param0.gen_decl()
        self.assertEqual("int", s)
        self.assertEqual("int", param0.typemap.name)
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

        self.assertEqual(
            todict.to_dict(r),
            {
                "declarator": {
                    "func": {
                        "name": "func",
                        "pointer": [{"ptr": "*"}],
                        "typemap_name": "int",
                    },
                    "params": [
                        {
                            "specifier": ["int"],
                            "typemap_name": "int",
                            "declarator": {
                                "typemap_name": "int",
                            },
                        }
                    ],
                    "typemap_name": "int",
                },
                "specifier": ["int"],
                "typemap_name": "int",
            },
        )

    def test_type_function_pointer2(self):
        """Function pointer
        """
        symtab = declast.SymbolTable()

        r = declast.check_decl("int *(*func)(int *arg)", symtab)

        s = r.gen_decl()
        self.assertEqual("int * ( * func)(int * arg)", s)

        self.assertEqual("int", r.typemap.name)
        self.assertEqual("func", r.name)
        self.assertTrue(r.is_pointer())
        self.assertFalse(r.is_reference())
        self.assertTrue(r.is_function_pointer())

        self.assertNotEqual(None, r.params)
        self.assertEqual(1, len(r.params))

        param0 = r.params[0]
        s = param0.gen_decl()
        self.assertEqual("int * arg", s)
        self.assertEqual("int", param0.typemap.name)
        self.assertEqual("arg", param0.name)
        self.assertTrue(param0.is_pointer())
        self.assertFalse(param0.is_reference())
        self.assertFalse(param0.is_function_pointer())

    # decl
    def test_decl01(self):
        """Simple declaration"""
        symtab = declast.SymbolTable()

        r = declast.check_decl("void foo", symtab)

        s = r.gen_decl()
        self.assertEqual("void foo", s)

        self.assertEqual(
            todict.to_dict(r),
            {
                "declarator": {
                    "name": "foo",
                    "typemap_name": "void",
                },
                "specifier": ["void"],
                "typemap_name": "void",
            },
        )
        self.assertEqual("foo", r.get_name())

    def test_decl02(self):
        """Simple declaration with attribute"""
        symtab = declast.SymbolTable()

        r = declast.check_decl("void foo +alias(junk)", symtab)

        s = r.gen_decl()
        self.assertEqual("void foo +alias(junk)", s)

        self.assertEqual(
            todict.to_dict(r),
            {
                "declarator": {
                    "name": "foo",
                    "attrs": {"alias": "junk"},
                    "typemap_name": "void",
                },
                "specifier": ["void"],
                "typemap_name": "void",
            },
        )
        self.assertEqual("foo", r.get_name())

    def test_decl03(self):
        """Empty parameter list"""
        symtab = declast.SymbolTable()

        r = declast.check_decl("void foo()", symtab)

        s = r.gen_decl()
        self.assertEqual("void foo(void)", s)

        self.assertEqual(
            todict.to_dict(r),
            {
                "declarator": {
                    "name": "foo",
                    "params": [],
                    "typemap_name": "void",
                },
                "specifier": ["void"],
                "typemap_name": "void",
            },
        )
        self.assertEqual("foo", r.get_name())
        self.assertEqual("void", r.typemap.name)
        self.assertFalse(r.is_pointer())
        self.assertEqual("subroutine", r.get_subprogram())
        self.assertIsNone(r.find_arg_by_name("argnone"))

    def test_decl04(self):
        """const method"""
        symtab = declast.SymbolTable()

        r = declast.check_decl("void *foo() const", symtab)

        s = r.gen_decl()
        self.assertEqual("void * foo(void) const", s)

        self.assertEqual(
            todict.to_dict(r),
            {
                "declarator": {
                    "name": "foo",
                    "pointer": [{"ptr": "*"}],
                    "params": [],
                    "func_const": True,
                    "typemap_name": "void",
                },
                "specifier": ["void"],
                "typemap_name": "void",
            },
        )
        self.assertEqual("foo", r.get_name())
        self.assertEqual("void", r.typemap.name)
        self.assertTrue(r.is_pointer())
        self.assertEqual("function", r.get_subprogram())

    def test_decl05(self):
        """Single argument"""
        symtab = declast.SymbolTable()

        r = declast.check_decl("void foo(int arg1)", symtab)

        s = r.gen_decl()
        self.assertEqual("void foo(int arg1)", s)

        self.assertEqual(
            todict.to_dict(r),
            {
                "declarator": {
                    "name": "foo",
                    "params": [
                        {
                            "declarator": {
                                "name": "arg1",
                                "typemap_name": "int",
                            },
                            "specifier": ["int"],
                            "typemap_name": "int",
                        }
                    ],
                    "typemap_name": "void",
                },
                "specifier": ["void"],
                "typemap_name": "void",
            },
        )
        self.assertEqual("foo", r.get_name())

    def test_decl06(self):
        """multiple arguments"""
        symtab = declast.SymbolTable()

        r = declast.check_decl("void foo(int arg1, double arg2)", symtab)

        s = r.gen_decl()
        self.assertEqual("void foo(int arg1, double arg2)", s)

        self.assertEqual(
            todict.to_dict(r),
            {
                "declarator": {
                    "name": "foo",
                    "params": [
                        {
                            "declarator": {
                                "name": "arg1",
                                "typemap_name": "int",
                            },
                            "specifier": ["int"],
                            "typemap_name": "int",
                        },
                        {
                            "declarator": {
                                "name": "arg2",
                                "typemap_name": "double",
                            },
                            "specifier": ["double"],
                            "typemap_name": "double",
                        },
                    ],
                    "typemap_name": "void",
                },
                "specifier": ["void"],
                "typemap_name": "void",
            },
        )
        self.assertEqual("foo", r.get_name())

        self.assertIsNotNone(r.find_arg_by_name("arg1"))
        self.assertIsNotNone(r.find_arg_by_name("arg2"))
        self.assertIsNone(r.find_arg_by_name("argnone"))

    def test_decl07(self):
        """Return string"""
        symtab = declast.SymbolTable()
        symtab.create_std_namespace()
        
        r = declast.check_decl("const std::string& getName() const", symtab)

        s = r.gen_decl()
        self.assertEqual("const std::string & getName(void) const", s)
        self.assertFalse(r.is_pointer())
        self.assertTrue(r.is_reference())
        self.assertEqual(1, r.is_indirect())

        self.assertEqual(
            todict.to_dict(r),
            {
                "const": True,
                "declarator": {
                    "name": "getName",
                    "func_const": True,
                    "pointer": [{"ptr": "&"}],
                    "params": [],
                    "typemap_name": "std::string",
                },
                "specifier": ["std::string"],
                "typemap_name": "std::string",
            },
        )
        self.assertEqual("getName", r.get_name())

    def test_decl08(self):
        """Test attributes.
        """
        symtab = declast.SymbolTable()

        r = declast.check_decl(
            "const void foo("
            "int arg1+in, double arg2+out)"
            "+len=30 +attr2(True)",
            symtab)

        s = r.gen_decl()
        self.assertEqual(
            "const void foo("
            "int arg1 +in, double arg2 +out)"
            " +attr2(True)+len(30)",
            s,
        )

        self.assertEqual(
            todict.to_dict(r),
            {
                "const": True,
                "declarator": {
                    "name": "foo",
                    "attrs": {"attr2": "True", "len": 30},
                    "params": [
                        {
                            "declarator": {
                                "name": "arg1",
                                "attrs": {"in": True},
                                "typemap_name": "int",
                            },
                            "specifier": ["int"],
                            "typemap_name": "int",
                        },
                        {
                            "declarator": {
                                "name": "arg2",
                                "attrs": {"out": True},
                                "typemap_name": "double",
                            },
                            "specifier": ["double"],
                            "typemap_name": "double",
                        },
                    ],
                    "typemap_name": "void",
                },
                "specifier": ["void"],
                "typemap_name": "void",
            },
        )
        self.assertEqual("foo", r.get_name())

    def test_decl09a(self):
        """Test constructor
        """
        symtab = declast.SymbolTable()
        declast.check_decl("class Class1", symtab)
        
        r = declast.check_decl("Class1()", symtab)

        s = r.gen_decl()
        self.assertEqual("Class1(void)", s)

        self.assertEqual(
            todict.to_dict(r),
            {
                "specifier": ["Class1"],
                "typemap_name": "Class1",
                'declarator': {
                    "attrs": {"_constructor": True, "_name": "ctor"},
                    "params": [],
                    "typemap_name": "Class1",
                },
            },
        )
        self.assertEqual("ctor", r.get_name())
        self.assertFalse(r.is_pointer())
        self.assertFalse(r.is_reference())
        # must provide the name since the ctor has no name
        # cxx_type and c_type are not defined yet
        self.assertEqual("--NOTYPE-- ctor(void)", r.gen_arg_as_cxx())
        self.assertEqual("--NOTYPE-- * ctor", r.gen_arg_as_c(params=None))

    def test_decl09b(self):
        """Test constructor +name
        """
        symtab = declast.SymbolTable()
        declast.check_decl("class Class1", symtab)

        r = declast.check_decl("Class1() +name(new)", symtab)

        s = r.gen_decl()
        self.assertEqual("Class1(void) +name(new)", s)

        self.assertEqual(
            todict.to_dict(r),
            {
                "specifier": ["Class1"],
                "typemap_name": "Class1",
                'declarator': {
                    "attrs": {"_constructor": True, "_name": "ctor", "name": "new"},
                    "params": [],
                    "typemap_name": "Class1",
                },
            },
        )
        self.assertEqual("new", r.get_name())
        self.assertFalse(r.is_pointer())
        self.assertFalse(r.is_reference())
        self.assertEqual(0, r.is_indirect())
        self.assertEqual("--NOTYPE-- new", r.gen_arg_as_cxx(params=None))
        self.assertEqual("--NOTYPE-- * new(void)", r.gen_arg_as_c())

    def test_decl09c(self):
        """Test destructor
        """
        symtab = declast.SymbolTable()
        declast.check_decl("class Class1", symtab)
        
        r = declast.check_decl("~Class1(void)", symtab)

        s = r.gen_decl()
        self.assertEqual("~Class1(void)", s)

        self.assertEqual(
            todict.to_dict(r),
            {
                "specifier": ["void"],
                "typemap_name": "void",
                'declarator': {
                    "attrs": {"_destructor": "Class1", "_name": "dtor"},
                    "params": [],
                    "typemap_name": "void",
                },
            },
        )
        self.assertEqual("dtor", r.get_name())
        self.assertFalse(r.is_pointer())
        self.assertFalse(r.is_reference())
        self.assertEqual(0, r.is_indirect())
        self.assertEqual("void dtor(void)", r.gen_arg_as_cxx())
        self.assertEqual("void dtor(void)", r.gen_arg_as_c())

    def test_inheritance0(self):
        symtab = declast.SymbolTable()

        declast.check_decl("class Class1", symtab);
        symtab.pop_scope()

        r2 = declast.check_decl("class Class2 : public Class1", symtab)
        self.assertIsInstance(r2, declast.Declaration)
        self.assertIsInstance(r2.class_specifier, declast.CXXClass)
        self.assertEqual("class Class2: public Class1", todict.print_node(r2))
        self.assertEqual(
            todict.to_dict(r2),
            {
                'class_specifier': {
                    'name': 'Class2',
                    'baseclass': [('public', 'Class1', 'Class1')]
                },
                'specifier': ['class Class2'],
                'typemap_name': 'Class2',
                "declarator": {
                    'typemap_name': 'Class2',
                },
            }
        )

        with self.assertRaises(RuntimeError) as context:
            r2 = declast.check_decl("class Class3 : public public", symtab)
        self.assertTrue("Expected ID, found PUBLIC" in str(context.exception))
        with self.assertRaises(RuntimeError) as context:
            r2 = declast.check_decl("class Class3 : public int", symtab)
        self.assertTrue("Expected ID, found TYPE_SPECIFIER" in str(context.exception))
        
    def test_inheritance(self):
        symtab = declast.SymbolTable()
        
        class1 = declast.check_decl("class Class1", symtab);
        symtab.pop_scope()

        self.assertIsInstance(class1, declast.Declaration)
        self.assertIsInstance(class1.class_specifier, declast.CXXClass)
        # XXX - base needs a typemap as the 3rd member.
        class2 = declast.check_decl("class Class2 : public Class1", symtab)
# GGG        class2 = declast.check_decl("class Class2", base=[("public", "Class1")])
        self.assertIsInstance(class2, declast.Declaration)
        self.assertIsInstance(class2.class_specifier, declast.CXXClass)

    def test_decl09d(self):
        """Return pointer to Class instance
        """
        symtab = declast.SymbolTable()
        declast.check_decl("class Class1", symtab)

        r = declast.check_decl("Class1 * make()", symtab)

        s = r.gen_decl()
        self.assertEqual("Class1 * make(void)", s)

        self.assertEqual(
            todict.to_dict(r),
            {
                "declarator": {
                    "name": "make",
                    "pointer": [{"ptr": "*"}],
                    "params": [],
                    "typemap_name": "Class1",
                },
                "specifier": ["Class1"],
                "typemap_name": "Class1",
            },
        )
        self.assertEqual("make", r.get_name())

    def test_decl10(self):
        """Test default arguments
        """
        symtab = declast.SymbolTable()
        symtab.create_std_namespace()

        r = declast.check_decl(
            "void name(int arg1 = 0, "
            "double arg2 = 0.0,"
            'std::string arg3 = "name",'
            "bool arg4 = true)",
            symtab
        )

        s = r.gen_decl()
        self.assertEqual(
            "void name(int arg1=0, "
            "double arg2=0.0, "
            'std::string arg3="name", '
            "bool arg4=true)",
            s,
        )

        self.assertEqual(
            todict.to_dict(r),
            {
                "declarator": {
                    "name": "name",
                    "params": [
                        {
                            "declarator": {
                                "name": "arg1",
                                "init": 0,
                                "typemap_name": "int",
                            },
                            "specifier": ["int"],
                            "typemap_name": "int",
                        },
                        {
                            "declarator": {
                                "name": "arg2",
                                "init": 0.0,
                                "typemap_name": "double",
                            },
                            "specifier": ["double"],
                            "typemap_name": "double",
                        },
                        {
                            "declarator": {
                                "name": "arg3",
                                "init": '"name"',
                                "typemap_name": "std::string",
                            },
                            "specifier": ["std::string"],
                            "typemap_name": "std::string",
                        },
                        {
                            "declarator": {
                                "name": "arg4",
                                "init": "true",
                                "typemap_name": "bool",
                            },
                            "specifier": ["bool"],
                            "typemap_name": "bool",
                        },
                    ],
                    "typemap_name": "void",
                },
                "specifier": ["void"],
                "typemap_name": "void",
            },
        )
        self.assertEqual("name", r.get_name())

    def test_decl11(self):
        """Test function template"""
        symtab = declast.SymbolTable()

        r = declast.check_decl("template<ArgType> void decl11(ArgType arg)",
                               symtab)

        # XXX - AttributeError: 'Template' object has no attribute 'gen_decl'
        s = r.decl.gen_decl()
        self.assertEqual("void decl11(ArgType arg)", s)

        self.assertEqual(
            todict.to_dict(r),
            {
                "decl": {
                    "declarator": {
                        "name": "decl11",
                        "params": [
                            {
                                "declarator": {
                                    "name": "arg",
                                },
                                "specifier": ["ArgType"],
                                "template_argument": "ArgType",
                            }
                        ],
                        "typemap_name": "void",
                    },
                    "specifier": ["void"],
                    "typemap_name": "void",
                },
                "parameters": [{"name": "ArgType"}],
            },
        )
        self.assertEqual("decl11", r.decl.get_name())

    def test_decl12(self):
        """Test templates
        Test std::string and string types.
        """
        symtab = declast.SymbolTable()
        symtab.create_std_namespace()
        symtab.using_directive("std")

        r = declast.check_decl(
            "void decl12(std::vector<std::string> arg1, string arg2)",
            symtab
        )

        s = r.gen_decl()
        self.assertEqual(
            "void decl12(std::vector<std::string> arg1, string arg2)", s
        )

        self.assertEqual(
            todict.to_dict(r),
            {
                "declarator": {
                    "name": "decl12",
                    "typemap_name": "void",
                    "params": [
                        {
                            "declarator": {
                                "name": "arg1",
                                "typemap_name": "std::vector",
                            },
                            "specifier": ["std::vector"],
                            "template_arguments": [
                                {
                                    "specifier": ["std::string"],
                                    "typemap_name": "std::string",
                                    "declarator": {
                                        "typemap_name": "std::string",
                                    },
                                }
                            ],
                            "typemap_name": "std::vector",
                        },
                        {
                            "declarator": {
                                "name": "arg2",
                                "typemap_name": "std::string",
                            },
                            "specifier": ["string"],
                            "typemap_name": "std::string",
                        },
                    ],
                },
                "specifier": ["void"],
                "typemap_name": "void",
            },
        )
        self.assertEqual("decl12", r.get_name())

    def test_decl13(self):
        """Test multi-specifier
        """
        symtab = declast.SymbolTable()

        r = declast.check_decl(
            "void decl13(" "long int arg1," "long long arg2," "unsigned int)",
            symtab
        )

        self.assertEqual(["long", "int"], r.params[0].specifier)
        self.assertEqual("long", r.params[0].typemap.name)

        self.assertEqual(["long", "long"], r.params[1].specifier)
        self.assertEqual("long_long", r.params[1].typemap.name)

        self.assertEqual(["unsigned", "int"], r.params[2].specifier)
        self.assertEqual("unsigned_int", r.params[2].typemap.name)

    def test_class_template(self):
        """Class templates"""
        symtab = declast.SymbolTable()

        r = declast.check_decl("template<typename T> class vector",
                               symtab)

        #        s = r.gen_decl()
        #        self.assertEqual("template<typename T> vector", s)

        self.assertEqual(
            todict.to_dict(r),
            {
                'decl': {
                    'class_specifier': {'name': 'vector'},
                    'specifier': ['class vector'],
                    'typemap_name': 'vector',
                    'declarator': {
                        'typemap_name': 'vector',
                    },
                },
                'parameters': [{'name': 'T'}]
            }
        )

        r = declast.check_decl("template<Key,T> class map", symtab)

        #        s = r.gen_decl()
        #        self.assertEqual("template<typename Key, typename T> map", s)

        self.assertEqual(
            todict.to_dict(r),
            {
                'decl': {
                    'class_specifier': {'name': 'map'},
                    'specifier': ['class map'],
                    'typemap_name': 'vector::map',
                    'declarator': {
                        'typemap_name': 'vector::map',
                    },
                },
                'parameters': [{'name': 'Key'}, {'name': 'T'}]
            }
        )

    def test_as_arg(self):
        symtab = declast.SymbolTable()
        symtab.create_std_namespace()
        
        r = declast.check_decl("const std::string& getName() const", symtab)

        s = r.gen_decl()
        self.assertEqual("const std::string & getName(void) const", s)

        r.result_as_arg("output")
        s = r.gen_decl()
        self.assertEqual("void getName(const std::string & output) const", s)

    def test_copy01(self):
        """Test copy"""
        symtab = declast.SymbolTable()
        symtab.create_std_namespace()
        
        r = declast.check_decl(
            "const std::string& Function4b("
            "const std::string& arg1,"
            "const std::string& arg2 )",
            symtab
        )
        self.assertTrue(r.is_reference())
        self.assertEqual(r.name, "Function4b")

        r2 = copy.deepcopy(r)

        r2.name = "newname"
        self.assertEqual(r.name, "Function4b")  # first is unchanged
        self.assertEqual(r2.name, "newname")

    def test_struct(self):
        symtab = declast.SymbolTable()
        
        struct = declast.check_decl("""
struct Cstruct_list {
    int nitems;
    int *ivalue;
};
""", symtab)
        self.assertIsInstance(struct, declast.Declaration)
        self.assertIsInstance(struct.class_specifier, declast.Struct)
        members = struct.class_specifier.members
        self.assertEqual(2, len(members))
        ast = members[0]
        self.assertEqual(
            todict.to_dict(ast), {
                'declarator': {
                    'name': 'nitems',
                    'typemap_name': 'int'
                },
                'specifier': ['int'],
                'typemap_name': 'int'
            })
        ast = members[1]
        self.assertEqual(
            todict.to_dict(ast), {
                'declarator': {
                    'name': 'ivalue',
                    'pointer': [{   'ptr': '*'}],
                    'typemap_name': 'int',
                },
                'specifier': ['int'],
                'typemap_name': 'int',
            })


class CheckExpr(unittest.TestCase):
    # No need for namespace

    def test_constant1(self):
        r = declast.check_expr("20")
        self.assertEqual("20", todict.print_node(r))
        self.assertEqual(
            {"constant": "20"},
            todict.to_dict(r))

    def test_identifier1(self):
        r = declast.check_expr("id")
        self.assertEqual("id", todict.print_node(r))
        self.assertEqual(
            {"name": "id"},
            todict.to_dict(r))

    def test_identifier_no_args(self):
        r = declast.check_expr("id()")
        self.assertEqual("id()", todict.print_node(r))
        self.assertEqual(
            {"name": "id", "args": []},
            todict.to_dict(r))

    def test_identifier_with_arg(self):
        r = declast.check_expr("id(arg1)")
        self.assertEqual("id(arg1)", todict.print_node(r))
        self.assertEqual(
            {"name": "id", "args": [{"name": "arg1"}]},
            todict.to_dict(r)
        )

    def test_identifier_with_args(self):
        r = declast.check_expr("id(arg1,1)")
        self.assertEqual("id(arg1,1)", todict.print_node(r))
        self.assertEqual(
            {"name": "id", "args": [
                {"name": "arg1"},
                {"constant": "1"},
            ]},
            todict.to_dict(r)
        )

    def test_constant(self):
        r = declast.check_expr("1 + 2.345")
        self.assertEqual("1+2.345", todict.print_node(r))
        self.assertEqual(
            {
                "left": {"constant": "1"},
                "op": "+",
                "right": {"constant": "2.345"},
            },
            todict.to_dict(r)
        )

    def test_binary(self):
        r = declast.check_expr("a + b * c")
        self.assertEqual("a+b*c", todict.print_node(r))
        self.assertEqual(
            {
                "left": {"name": "a"},
                "op": "+",
                "right": {
                    "left": {"name": "b"},
                    "op": "*",
                    "right": {"name": "c"},
                },
            },
            todict.to_dict(r),
        )

        r = declast.check_expr("(a + b) * c")
        self.assertEqual("(a+b)*c", todict.print_node(r))
        self.assertEqual(
            {
                "left": {
                    "node": {
                        "left": {"name": "a"},
                        "op": "+",
                        "right": {"name": "b"},
                    }
                },
                "op": "*",
                "right": {"name": "c"},
            },
            todict.to_dict(r),
        )

    def test_others(self):
        e = "size+2"
        r = declast.check_expr(e)
        self.assertEqual(e, todict.print_node(r))


class CheckNamespace(unittest.TestCase):
    def test_decl_namespace(self):
        """Parse a namespace"""
        symtab = declast.SymbolTable()
        r = declast.check_decl("namespace ns1", symtab)
        self.assertEqual("namespace ns1", todict.print_node(r))
        self.assertEqual(todict.to_dict(r), {"name": "ns1"})


class CheckTypedef(unittest.TestCase):
    def test_typedef1(self):
        symtab = declast.SymbolTable()
        r = declast.check_decl("typedef int TypeID;", symtab)
        self.assertEqual("typedef int TypeID", r.gen_decl())
        self.assertDictEqual(
            todict.to_dict(r),
            {
                "declarator": {
                    "name": "TypeID",
                    "typemap_name": "int",
                },
                "specifier": ["int"],
                "storage": ["typedef"],
                "typemap_name": "TypeID",
            },
        )

    def test_typedef2(self):
        symtab = declast.SymbolTable()
        parent = symtab.current
        r = declast.check_decl("typedef int TD2;", symtab)
        self.assertIn("TD2", parent.symbols)

        ntypemap = symtab.lookup_typemap("TD2")
        self.assertIsNotNone(ntypemap)
        self.assertEqual("TD2", ntypemap.name)
        self.assertEqual("TD2", ntypemap.cxx_type)
        self.assertEqual("int", ntypemap.typedef)

    def test_typedef_errors(self):
        symtab = declast.SymbolTable()
        with self.assertRaises(RuntimeError) as context:
            r = declast.check_decl("typedef none TypeID;", symtab)
        self.assertTrue(
            "Expected TYPE_SPECIFIER, found ID 'none'" in str(context.exception)
        )

    def test_typedef_fcnptr(self):
        symtab = declast.SymbolTable()
        node = declast.check_decl("typedef int(*func)();", symtab)
        self.assertIsInstance(node, declast.Declaration)
#        self.assertIsInstance(node, declast.Typedef) # GGG Typedef object?
        self.assertIn("func", symtab.current.symbols)


class CheckEnum(unittest.TestCase):
    def xxsetUp(self):
        self.symtab = declast.SymbolTable()

    def test_enum1(self):
        symtab = declast.SymbolTable()
        r = declast.check_decl("enum Color{RED=1,BLUE,WHITE}", symtab)
        self.assertEqual(
            "enum Color { RED = 1, BLUE, WHITE };", todict.print_node(r)
        )
        self.assertEqual(
            todict.to_dict(r),
            {
                'enum_specifier': {
                    'name': 'Color',
                    'members': [
                        {'name': 'RED', 'value': {'constant': '1'}},
                        {'name': 'BLUE'},
                        {'name': 'WHITE'}
                    ],
                },
                'specifier': ['enum Color'],
                "declarator": {
                    'typemap_name': 'Color',
                },
                'typemap_name': 'Color',
            }
        )

    def test_enum2(self):
        # enum trailing comma
        symtab = declast.SymbolTable()
        r = declast.check_decl("enum Color{RED=1,BLUE,WHITE,}", symtab)
        self.assertEqual(
            "enum Color { RED = 1, BLUE, WHITE };", todict.print_node(r)
        )

    def test_enum_var1_cxx(self):
        """declare a enum variable"""
        symtab = declast.SymbolTable()
        r = declast.check_decl("enum Color{RED=1,BLUE,WHITE,}", symtab)
        r = declast.check_decl("enum Color var;", symtab)
        self.assertEqual(
            "enum Color var", todict.print_node(r)
        )
        r = declast.check_decl("Color var2;", symtab)
        self.assertEqual(
            "Color var2", todict.print_node(r)
        )

    def test_enum_var1_c(self):
        """declare a enum variable"""
        symtab = declast.SymbolTable()
        r = declast.check_decl("enum Color{RED=1,BLUE,WHITE,}", symtab)
        r = declast.check_decl("enum Color var;", symtab)
        self.assertEqual(
            "enum Color var", todict.print_node(r)
        )


class CheckStruct(unittest.TestCase):
    def test_struct1(self):
        symtab = declast.SymbolTable()
        
        r = declast.check_decl("struct struct1 { int i; double d; };",
                               symtab)
        self.assertEqual(
            todict.to_dict(r),
            {
                'class_specifier': {
                    'members': [
                        {
                            'declarator': {
                                'name': 'i',
                                'typemap_name': 'int'
                            },
                            'specifier': ['int'],
                            'typemap_name': 'int'
                        },{
                            'declarator': {
                                'name': 'd',
                                'typemap_name': 'double',
                            },
                            'specifier': ['double'],
                            'typemap_name': 'double',
                        },
                    ],
                    'name': 'struct1',
                    'typemap_name': 'struct1'
                },
                'specifier': ['struct struct1'],
                "declarator": {
                    'typemap_name': 'struct1',
                },
                'typemap_name': 'struct1',
            }
        )


class CheckClass(unittest.TestCase):
    def test_class1(self):
        symtab = declast.SymbolTable()
        r = declast.check_decl("class Class1", symtab)
        self.assertIsInstance(r, declast.Declaration)
        self.assertIsInstance(r.class_specifier, declast.CXXClass)
        self.assertEqual("class Class1", todict.print_node(r))
        self.assertEqual(
            todict.to_dict(r),
            {
                'class_specifier': {'name': 'Class1'},
                'specifier': ['class Class1'],
                'declarator': {
                    'typemap_name': 'Class1',
                },
                'typemap_name': 'Class1',
            }
        )

    def test_class2(self):
        """Forward declare class in a library"""
        symtab = declast.SymbolTable()
        parent = symtab.current
        r = declast.check_decl("class Class1", symtab)
        self.assertIn("Class1", parent.symbols)
        self.assertIsInstance(parent.symbols["Class1"], declast.CXXClass)

        ntypemap = symtab.lookup_typemap("Class1")
        self.assertIsNotNone(ntypemap)
        self.assertEqual("Class1", ntypemap.name)
        self.assertEqual(None, ntypemap.cxx_type)
        self.assertEqual("shadow", ntypemap.base)

    def Xtest_class2_node(self):  # GGG move to ast
        """Add a class with declarations to a library"""
        library = ast.LibraryNode()
        library.add_declaration(
            "class Class1;", declarations=[dict(decl="void func1()")]
        )
        self.assertIn("Class1", library.symbols)
        sym = library.symbols["Class1"]
        self.assertIsInstance(sym, ast.ClassNode)
        self.assertIs(sym, library.classes[0])

    def test_class_in_namespace(self):
        """Forward declare a class in a namespace"""
        symtab = declast.SymbolTable()
        ns = declast.check_decl("namespace ns", symtab)
        r = declast.check_decl("class Class2", symtab)
        self.assertIn("Class2", ns.symbols)
        self.assertIsInstance(ns.symbols["Class2"], declast.CXXClass)

        ntypemap = symtab.lookup_typemap("ns::Class2")
        self.assertIsNotNone(ntypemap)
        self.assertEqual("ns::Class2", ntypemap.name)
        self.assertEqual(None, ntypemap.cxx_type)


if __name__ == "__main__":
    unittest.main()
