# Copyright (c) 2017-2020, Lawrence Livermore National Security, LLC and
# other Shroud Project Developers.
# See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
########################################################################
"""
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

# Useful to format reference output of to_dict
#import pprint
#pp = pprint.PrettyPrinter(indent=4)
#        print(pp.pprint(todict.to_dict(r)))

class CheckParse(unittest.TestCase):
    maxDiff = None

    def setUp(self):
        self.library = ast.LibraryNode(library="cc")
        self.class1 = self.library.add_class("Class1")

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
        self.assertEqual("scalar", r.get_indirect_stmt())
        self.assertEqual(None, r.get_array_size())

        r = declast.check_decl("int var1")
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

        r = declast.check_decl("const int var1")
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
                    'pointer': []},
                'specifier': ['int'],
                'typemap_name': 'int'})
        self.assertEqual("scalar", r.get_indirect_stmt())

        r = declast.check_decl("int const var1")
        s = r.gen_decl()
        self.assertEqual("const int var1", s)
        self.assertEqual(
            todict.to_dict(r), {
                'const': True,
                'declarator': {
                    'name': 'var1',
                    'pointer': []},
                'specifier': ['int'],
                'typemap_name': 'int'})
        self.assertEqual("scalar", r.get_indirect_stmt())
        r = declast.check_decl("int *var1 +dimension(:)")
        self.assertIsNone(r.get_subprogram())
        self.assertEqual(1, r.is_pointer())
        s = r.gen_decl()
        self.assertEqual("int * var1 +dimension(:)", s)
        self.assertEqual("int * var1", r.gen_arg_as_c())
        self.assertEqual("int var1", r.gen_arg_as_c(as_scalar=True))
        self.assertEqual("int * var1", r.gen_arg_as_cxx())
        self.assertEqual("integer(C_INT) :: var1(:)", r.gen_arg_as_fortran())
        self.assertEqual("integer(C_INT) :: var1(*)", r.bind_c())

        r = declast.check_decl("const int * var1")
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

        r = declast.check_decl("int * const var1")
        s = r.gen_decl()
        self.assertEqual("int * const var1", s)
        self.assertEqual(
            todict.to_dict(r), {
                'declarator': {
                    'name': 'var1', 'pointer': [
                        {'const': True, 'ptr': '*'}]},
                'specifier': ['int'],
                'typemap_name': 'int'})
        self.assertEqual("*", r.get_indirect_stmt())
        self.assertEqual(None, r.get_array_size())

        r = declast.check_decl("int **var1")
        s = r.gen_decl()
        self.assertEqual("int * * var1", s)
        self.assertEqual(
            todict.to_dict(r), {
                'declarator': {
                    'name': 'var1', 'pointer': [
                        {'ptr': '*'}, {'ptr': '*'}]},
                'specifier': ['int'],
                'typemap_name': 'int'
            })
        self.assertEqual("**", r.get_indirect_stmt())
        self.assertEqual(None, r.get_array_size())

        r = declast.check_decl("int &*var1")
        s = r.gen_decl()
        self.assertEqual("int & * var1", s)
        self.assertEqual(
            todict.to_dict(r), {
                'declarator': {
                    'name': 'var1',
                    'pointer': [{   'ptr': '&'}, {   'ptr': '*'}]
                },
                'specifier': ['int'],
                'typemap_name': 'int'
            })
        self.assertEqual("&*", r.get_indirect_stmt())
        self.assertEqual("int **", r.as_cast())

        r = declast.check_decl("const int * const * const var1")
        s = r.gen_decl()
        self.assertEqual("const int * const * const var1", s)
        self.assertEqual(
            todict.to_dict(r), {
                'const': True,
                'declarator': {
                    'name': 'var1',
                      'pointer': [
                          {   'const': True, 'ptr': '*'},
                          {   'const': True, 'ptr': '*'}]},
                'specifier': ['int'],
                'typemap_name': 'int'
            })
        self.assertEqual("**", r.get_indirect_stmt())
        self.assertEqual("int **", r.as_cast())

        r = declast.check_decl("long long var2")
        s = r.gen_decl()
        self.assertEqual("long long var2", s)

        # test attributes
        r = declast.check_decl("int m_ivar +readonly +name(ivar)")
        self.assertEqual(
            todict.to_dict(r),
            {
                "attrs": {"name": "ivar", "readonly": True},
                "declarator": {"name": "m_ivar", "pointer": []},
                "specifier": ["int"],
                "typemap_name": "int",
            },
        )

    def test_type_int_array(self):
        r = declast.check_decl("int var1[20]")
        self.assertEqual("int var1[20]", str(r))
        s = r.gen_decl()
        self.assertEqual("int var1[20]", s)
        self.assertEqual(
            {'declarator': {
                'name': 'var1', 'pointer': []},
             'array': [{ 'constant': '20'}],
             'specifier': ['int'],
             'typemap_name': 'int'},
            todict.to_dict(r)
        )
        self.assertEqual("int *", r.as_cast())
        self.assertEqual(
            "int var1[20]",
            r.gen_arg_as_c())
        self.assertEqual(
            "integer(C_INT) :: var1(20)",
            r.gen_arg_as_fortran())
        
        r = declast.check_decl("int var2[20][10]")
        self.assertEqual("int var2[20][10]", str(r))
        s = r.gen_decl()
        self.assertEqual("int var2[20][10]", s)
        self.assertEqual(
            {'declarator': {
                'name': 'var2', 'pointer': []},
             'array': [
                 { 'constant': '20'},
                 { 'constant': '10'},
             ],
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
        
        r = declast.check_decl("int var3[DEFINE + 3]")
        self.assertEqual("int var3[DEFINE+3]", str(r))
        s = r.gen_decl()
        self.assertEqual("int var3[DEFINE+3]", s)
        self.assertEqual(
            {'array': [
                {'left': {   'name': 'DEFINE'},
                 'op': '+',
                 'right': {   'constant': '3'}}],
             'declarator': {   'name': 'var3', 'pointer': []},
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
        typemap.initialize()

        r = declast.check_decl("char var1")
        s = r.gen_decl()
        self.assertEqual("char var1", s)
        self.assertEqual("char", r.as_cast())

        r = declast.check_decl("char *var1")
        self.assertEqual("char *", r.as_cast())
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
        self.assertEqual(2, r.is_indirect())
        self.assertEqual(2, r.is_array())
        self.assertEqual('**', r.get_indirect_stmt())
        self.assertEqual("char **", r.as_cast())
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
        self.assertEqual("char *", r.as_cast())

        r = declast.check_decl("std::string &var1")
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
        r = declast.check_decl("char var1[20]")
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
                'name': 'var1', 'pointer': []},
             'array': [{ 'constant': '20'}],
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
        
        r = declast.check_decl("char var2[20][10][5]")
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
                'name': 'var2', 'pointer': []},
             'array': [
                 { 'constant': '20'},
                 { 'constant': '10'},
                 { 'constant': '5'},
             ],
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
        
        r = declast.check_decl("char var3[DEFINE + 3]")
        self.assertEqual(0, r.is_indirect())
        self.assertEqual(1, r.is_array())
        self.assertEqual("char *", r.as_cast())
        self.assertEqual('[]', r.get_indirect_stmt())
        self.assertEqual("DEFINE+3", r.get_array_size())
        self.assertEqual("char var3[DEFINE+3]", str(r))
        s = r.gen_decl()
        self.assertEqual("char var3[DEFINE+3]", s)
        self.assertEqual(
            {'array': [
                {'left': {   'name': 'DEFINE'},
                 'op': '+',
                 'right': {   'constant': '3'}}],
             'declarator': {   'name': 'var3', 'pointer': []},
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
    
        r = declast.check_decl("char *var4[44]")
        self.assertEqual("char * var4[44]", str(r))
        self.assertEqual(1, r.is_indirect())
        self.assertEqual(2, r.is_array())
        self.assertEqual("char **", r.as_cast())
        self.assertEqual('*[]', r.get_indirect_stmt())
        self.assertEqual("44", r.get_array_size())
        s = r.gen_decl()
        self.assertEqual("char * var4[44]", s)
        self.assertEqual(
            {'array': [{'constant': '44'}],
             'declarator': {   'name': 'var4',
                               'pointer': [{'ptr': '*'}]},
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
        r = declast.check_decl("std::vector<int> var1")
        s = r.gen_decl()
        self.assertEqual("std::vector<int> var1", s)
        self.assertEqual(
            todict.to_dict(r),
            {
                "declarator": {"name": "var1", "pointer": []},
                "specifier": ["std::vector"],
                "template_arguments": [
                    {"specifier": ["int"], "typemap_name": "int"}
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

        r = declast.check_decl("std::vector<long long> var1")
        s = r.gen_decl()
        self.assertEqual("std::vector<long long> var1", s)
        self.assertEqual(
            todict.to_dict(r),
            {
                "declarator": {"name": "var1", "pointer": []},
                "specifier": ["std::vector"],
                "template_arguments": [
                    {"specifier": ["long", "long"], "typemap_name": "long_long"}
                ],
                "typemap_name": "std::vector",
            },
        )

        r = declast.check_decl("std::vector<std::string> var1")
        s = r.gen_decl()
        self.assertEqual("std::vector<std::string> var1", s)
        self.assertEqual(
            todict.to_dict(r),
            {
                "declarator": {"name": "var1", "pointer": []},
                "specifier": ["std::vector"],
                "template_arguments": [
                    {
                        "specifier": ["std::string"],
                        "typemap_name": "std::string",
                    }
                ],
                "typemap_name": "std::vector",
            },
        )

    def test_template_argument_list(self):
        decl = "<int>"
        parser = declast.Parser(decl, None)
        r = parser.template_argument_list()
        self.assertEqual(
            todict.to_dict(r), [{"specifier": ["int"], "typemap_name": "int"}]
        )

        # self.library creates a global namespace with std::string
        decl = "<std::string, int>"
        parser = declast.Parser(decl, self.library)
        r = parser.template_argument_list()
        self.assertEqual(
            todict.to_dict(r),
            [
                {"specifier": ["std::string"], "typemap_name": "std::string"},
                {"specifier": ["int"], "typemap_name": "int"},
            ],
        )

    def test_declaration_specifier_error(self):
        with self.assertRaises(RuntimeError) as context:
            declast.check_decl("none var1")
        self.assertTrue(
            "Expected TYPE_SPECIFIER, found ID 'none'" in str(context.exception)
        )

        with self.assertRaises(RuntimeError) as context:
            declast.check_decl("std::int var1")
        self.assertTrue(
            "Expected ID, found TYPE_SPECIFIER" in str(context.exception)
        )

        with self.assertRaises(RuntimeError) as context:
            declast.check_decl("std::none var1")
        self.assertTrue(
            "Symbol 'none' is not in namespace 'std'" in str(context.exception)
        )

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
        r = declast.check_decl("int (*func)(int)")

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
                    "func": {"name": "func", "pointer": [{"ptr": "*"}]},
                    "pointer": [],
                },
                "params": [{"specifier": ["int"], "typemap_name": "int"}],
                "specifier": ["int"],
                "typemap_name": "int",
            },
        )

    def test_type_function_pointer2(self):
        """Function pointer
        """
        r = declast.check_decl("int *(*func)(int *arg)")

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
        r = declast.check_decl("void foo")

        s = r.gen_decl()
        self.assertEqual("void foo", s)

        self.assertEqual(
            todict.to_dict(r),
            {
                "declarator": {"name": "foo", "pointer": []},
                "specifier": ["void"],
                "typemap_name": "void",
            },
        )
        self.assertEqual("foo", r.get_name())

    def test_decl02(self):
        """Simple declaration with attribute"""
        r = declast.check_decl("void foo +alias(junk)")

        s = r.gen_decl()
        self.assertEqual("void foo +alias(junk)", s)

        self.assertEqual(
            todict.to_dict(r),
            {
                "attrs": {"alias": "junk"},
                "declarator": {"name": "foo", "pointer": []},
                "specifier": ["void"],
                "typemap_name": "void",
            },
        )
        self.assertEqual("foo", r.get_name())

    def test_decl03(self):
        """Empty parameter list"""
        r = declast.check_decl("void foo()")

        s = r.gen_decl()
        self.assertEqual("void foo()", s)

        self.assertEqual(
            todict.to_dict(r),
            {
                "declarator": {"name": "foo", "pointer": []},
                "params": [],
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
        r = declast.check_decl("void *foo() const")

        s = r.gen_decl()
        self.assertEqual("void * foo() const", s)

        self.assertEqual(
            todict.to_dict(r),
            {
                "declarator": {"name": "foo", "pointer": [{"ptr": "*"}]},
                "func_const": True,
                "params": [],
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
        r = declast.check_decl("void foo(int arg1)")

        s = r.gen_decl()
        self.assertEqual("void foo(int arg1)", s)

        self.assertEqual(
            todict.to_dict(r),
            {
                "declarator": {"name": "foo", "pointer": []},
                "params": [
                    {
                        "declarator": {"name": "arg1", "pointer": []},
                        "specifier": ["int"],
                        "typemap_name": "int",
                    }
                ],
                "specifier": ["void"],
                "typemap_name": "void",
            },
        )
        self.assertEqual("foo", r.get_name())

    def test_decl06(self):
        """multiple arguments"""
        r = declast.check_decl("void foo(int arg1, double arg2)")

        s = r.gen_decl()
        self.assertEqual("void foo(int arg1, double arg2)", s)

        self.assertEqual(
            todict.to_dict(r),
            {
                "declarator": {"name": "foo", "pointer": []},
                "params": [
                    {
                        "declarator": {"name": "arg1", "pointer": []},
                        "specifier": ["int"],
                        "typemap_name": "int",
                    },
                    {
                        "declarator": {"name": "arg2", "pointer": []},
                        "specifier": ["double"],
                        "typemap_name": "double",
                    },
                ],
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
        r = declast.check_decl("const std::string& getName() const")

        s = r.gen_decl()
        self.assertEqual("const std::string & getName() const", s)
        self.assertFalse(r.is_pointer())
        self.assertTrue(r.is_reference())
        self.assertEqual(1, r.is_indirect())

        self.assertEqual(
            todict.to_dict(r),
            {
                "const": True,
                "declarator": {"name": "getName", "pointer": [{"ptr": "&"}]},
                "func_const": True,
                "params": [],
                "specifier": ["std::string"],
                "typemap_name": "std::string",
            },
        )
        self.assertEqual("getName", r.get_name())

    def test_decl08(self):
        """Test attributes.
        """
        r = declast.check_decl(
            "const void foo("
            "int arg1+in, double arg2+out)"
            "+len=30 +attr2(True)"
        )

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
                "attrs": {"attr2": "True", "len": 30},
                "const": True,
                "declarator": {"name": "foo", "pointer": []},
                "params": [
                    {
                        "attrs": {"in": True},
                        "declarator": {"name": "arg1", "pointer": []},
                        "specifier": ["int"],
                        "typemap_name": "int",
                    },
                    {
                        "attrs": {"out": True},
                        "declarator": {"name": "arg2", "pointer": []},
                        "specifier": ["double"],
                        "typemap_name": "double",
                    },
                ],
                "specifier": ["void"],
                "typemap_name": "void",
            },
        )
        self.assertEqual("foo", r.get_name())

    def test_decl09a(self):
        """Test constructor
        """
        r = declast.check_decl("Class1()", namespace=self.class1)

        s = r.gen_decl()
        self.assertEqual("Class1()", s)

        self.assertEqual(
            todict.to_dict(r),
            {
                "attrs": {"_constructor": True, "_name": "ctor"},
                "params": [],
                "specifier": ["Class1"],
                "typemap_name": "Class1",
            },
        )
        self.assertEqual("ctor", r.get_name())
        self.assertFalse(r.is_pointer())
        self.assertFalse(r.is_reference())
        # must provide the name since the ctor has no name
        self.assertEqual("Class1 ctor()", r.gen_arg_as_cxx())
        self.assertEqual("CC_Class1 ctor", r.gen_arg_as_c(params=None))

    def test_decl09b(self):
        """Test constructor +name
        """
        r = declast.check_decl("Class1() +name(new)", namespace=self.class1)

        s = r.gen_decl()
        self.assertEqual("Class1() +name(new)", s)

        self.assertEqual(
            todict.to_dict(r),
            {
                "attrs": {"_constructor": True, "_name": "ctor", "name": "new"},
                "params": [],
                "specifier": ["Class1"],
                "typemap_name": "Class1",
            },
        )
        self.assertEqual("new", r.get_name())
        self.assertFalse(r.is_pointer())
        self.assertFalse(r.is_reference())
        self.assertEqual(0, r.is_indirect())
        self.assertEqual("Class1 new", r.gen_arg_as_cxx(params=None))
        self.assertEqual("CC_Class1 new()", r.gen_arg_as_c())

    def test_decl09c(self):
        """Test destructor
        """
        r = declast.check_decl("~Class1()", namespace=self.class1)

        s = r.gen_decl()
        self.assertEqual("~Class1()", s)

        self.assertEqual(
            todict.to_dict(r),
            {
                "attrs": {"_destructor": True, "_name": "dtor"},
                "params": [],
                "specifier": ["Class1"],
                "typemap_name": "Class1",
            },
        )
        self.assertEqual("dtor", r.get_name())
        self.assertFalse(r.is_pointer())
        self.assertFalse(r.is_reference())
        self.assertEqual(0, r.is_indirect())
        self.assertEqual("Class1 dtor()", r.gen_arg_as_cxx())
        self.assertEqual("CC_Class1 dtor()", r.gen_arg_as_c())

    def test_decl09d(self):
        """Return pointer to Class instance
        """
        r = declast.check_decl("Class1 * make()", namespace=self.class1)

        s = r.gen_decl()
        self.assertEqual("Class1 * make()", s)

        self.assertEqual(
            todict.to_dict(r),
            {
                "declarator": {"name": "make", "pointer": [{"ptr": "*"}]},
                "params": [],
                "specifier": ["Class1"],
                "typemap_name": "Class1",
            },
        )
        self.assertEqual("make", r.get_name())

    def test_decl10(self):
        """Test default arguments
        """
        r = declast.check_decl(
            "void name(int arg1 = 0, "
            "double arg2 = 0.0,"
            'std::string arg3 = "name",'
            "bool arg4 = true)"
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
                "declarator": {"name": "name", "pointer": []},
                "params": [
                    {
                        "declarator": {"name": "arg1", "pointer": []},
                        "init": 0,
                        "specifier": ["int"],
                        "typemap_name": "int",
                    },
                    {
                        "declarator": {"name": "arg2", "pointer": []},
                        "init": 0.0,
                        "specifier": ["double"],
                        "typemap_name": "double",
                    },
                    {
                        "declarator": {"name": "arg3", "pointer": []},
                        "init": '"name"',
                        "specifier": ["std::string"],
                        "typemap_name": "std::string",
                    },
                    {
                        "declarator": {"name": "arg4", "pointer": []},
                        "init": "true",
                        "specifier": ["bool"],
                        "typemap_name": "bool",
                    },
                ],
                "specifier": ["void"],
                "typemap_name": "void",
            },
        )
        self.assertEqual("name", r.get_name())

    def test_decl11(self):
        """Test function template"""
        r = declast.check_decl("template<ArgType> void decl11(ArgType arg)")

        # XXX - AttributeError: 'Template' object has no attribute 'gen_decl'
        s = r.decl.gen_decl()
        self.assertEqual("void decl11(ArgType arg)", s)

        self.assertEqual(
            todict.to_dict(r),
            {
                "decl": {
                    "declarator": {"name": "decl11", "pointer": []},
                    "params": [
                        {
                            "declarator": {"name": "arg", "pointer": []},
                            "specifier": ["ArgType"],
                            "typemap_name": "ArgType",
                        }
                    ],
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
        r = declast.check_decl(
            "void decl12(std::vector<std::string> arg1, string arg2)"
        )

        s = r.gen_decl()
        self.assertEqual(
            "void decl12(std::vector<std::string> arg1, string arg2)", s
        )

        self.assertEqual(
            todict.to_dict(r),
            {
                "declarator": {"name": "decl12", "pointer": []},
                "params": [
                    {
                        "declarator": {"name": "arg1", "pointer": []},
                        "specifier": ["std::vector"],
                        "template_arguments": [
                            {
                                "specifier": ["std::string"],
                                "typemap_name": "std::string",
                            }
                        ],
                        "typemap_name": "std::vector",
                    },
                    {
                        "declarator": {"name": "arg2", "pointer": []},
                        "specifier": ["string"],
                        "typemap_name": "std::string",
                    },
                ],
                "specifier": ["void"],
                "typemap_name": "void",
            },
        )
        self.assertEqual("decl12", r.get_name())

    def test_decl13(self):
        """Test multi-specifier
        """
        r = declast.check_decl(
            "void decl13(" "long int arg1," "long long arg2," "unsigned int)"
        )

        self.assertEqual(["long", "int"], r.params[0].specifier)
        self.assertEqual("long", r.params[0].typemap.name)

        self.assertEqual(["long", "long"], r.params[1].specifier)
        self.assertEqual("long_long", r.params[1].typemap.name)

        self.assertEqual(["unsigned", "int"], r.params[2].specifier)
        self.assertEqual("unsigned_int", r.params[2].typemap.name)

    def test_class_template(self):
        """Class templates"""
        r = declast.check_decl("template<typename T> class vector")

        #        s = r.gen_decl()
        #        self.assertEqual("template<typename T> vector", s)

        self.assertEqual(
            todict.to_dict(r),
            {"decl": {"name": "vector"}, "parameters": [{"name": "T"}]},
        )

        r = declast.check_decl("template<Key,T> class map")

        #        s = r.gen_decl()
        #        self.assertEqual("template<typename Key, typename T> map", s)

        self.assertEqual(
            todict.to_dict(r),
            {
                "decl": {"name": "map"},
                "parameters": [{"name": "Key"}, {"name": "T"}],
            },
        )

    def test_as_arg(self):
        r = declast.check_decl("const std::string& getName() const")

        s = r.gen_decl()
        self.assertEqual("const std::string & getName() const", s)

        r.result_as_arg("output")
        s = r.gen_decl()
        self.assertEqual("void getName(const std::string & output) const", s)

    def test_copy01(self):
        """Test copy"""
        r = declast.check_decl(
            "const std::string& Function4b("
            "const std::string& arg1,"
            "const std::string& arg2 )"
        )
        self.assertTrue(r.is_reference())
        self.assertEqual(r.name, "Function4b")

        r2 = copy.deepcopy(r)

        r2.name = "newname"
        self.assertEqual(r.name, "Function4b")  # first is unchanged
        self.assertEqual(r2.name, "newname")

    def test_struct(self):
        struct = self.library.add_struct("""
struct Cstruct_list {
    int nitems;
    int *ivalue;
};
""")
        self.assertEqual(2, len(struct.variables))
        ast = struct.variables[0].ast
        self.assertEqual(
            todict.to_dict(ast), {
                'declarator': {
                    'name': 'nitems',
                    'pointer': []
                },
                'specifier': ['int'],
                'typemap_name': 'int'
            })
        ast = struct.variables[1].ast
        self.assertEqual(
            todict.to_dict(ast), {
                'declarator': {
                    'name': 'ivalue',
                    'pointer': [{   'ptr': '*'}]
                },
                'specifier': ['int'],
                'typemap_name': 'int'
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
        r = declast.check_decl("namespace ns1")
        self.assertEqual("namespace ns1", todict.print_node(r))
        self.assertEqual(todict.to_dict(r), {"name": "ns1"})


class CheckTypedef(unittest.TestCase):
    def XXXsetUp(self):
        library = ast.LibraryNode()

    def test_typedef1(self):
        r = declast.check_decl("typedef int TypeID;")
        self.assertEqual("typedef int TypeID", r.gen_decl())
        self.assertDictEqual(
            todict.to_dict(r),
            {
                "declarator": {"name": "TypeID", "pointer": []},
                "specifier": ["int"],
                "storage": ["typedef"],
                "typemap_name": "int",
            },
        )

    def test_typedef2(self):
        library = ast.LibraryNode()
        library.add_declaration("typedef int TD2;")
        self.assertIn("TD2", library.symbols)

        typedef = typemap.lookup_type("TD2")
        self.assertIsNotNone(typedef)
        self.assertEqual("TD2", typedef.name)
        self.assertEqual("TD2", typedef.cxx_type)
        self.assertEqual("int", typedef.typedef)

    def test_typedef_errors(self):
        with self.assertRaises(RuntimeError) as context:
            r = declast.check_decl("typedef none TypeID;")
        self.assertTrue(
            "Expected TYPE_SPECIFIER, found ID 'none'" in str(context.exception)
        )

        library = ast.LibraryNode()
        with self.assertRaises(NotImplementedError) as context:
            library.add_declaration("typedef int * TD2;")
        self.assertTrue(
            "Pointers not supported in typedef" in str(context.exception)
        )

        with self.assertRaises(NotImplementedError) as context:
            library.add_declaration("typedef int(*func)();")
        self.assertTrue(
            "Function pointers not supported in typedef"
            in str(context.exception)
        )


class CheckEnum(unittest.TestCase):
    def XXXsetUp(self):
        library = ast.LibraryNode()

    def test_enum1(self):
        r = declast.check_decl("enum Color{RED=1,BLUE,WHITE}")
        self.assertEqual(
            "enum Color { RED = 1, BLUE, WHITE };", todict.print_node(r)
        )
        self.assertEqual(
            todict.to_dict(r),
            {
                "name": "Color",
                "members": [
                    {"name": "RED", "value": {"constant": "1"}},
                    {"name": "BLUE"},
                    {"name": "WHITE"},
                ],
            },
        )

    def test_enum2(self):
        # enum trailing comma
        r = declast.check_decl("enum Color{RED=1,BLUE,WHITE,}")
        self.assertEqual(
            "enum Color { RED = 1, BLUE, WHITE };", todict.print_node(r)
        )


class CheckStruct(unittest.TestCase):
    def test_struct1(self):
        r = declast.check_decl("struct struct1 { int i; double d; };")
        self.assertEqual(
            "struct struct1 { int i;double d; };", todict.print_node(r)
        )
        self.assertEqual(
            todict.to_dict(r),
            {
                "members": [
                    {
                        "declarator": {"name": "i", "pointer": []},
                        "specifier": ["int"],
                        "typemap_name": "int",
                    },
                    {
                        "declarator": {"name": "d", "pointer": []},
                        "specifier": ["double"],
                        "typemap_name": "double",
                    },
                ],
                "name": "struct1",
            },
        )


class CheckClass(unittest.TestCase):
    def XXXsetUp(self):
        library = ast.LibraryNode()

    def test_class1(self):
        r = declast.check_decl("class Class1")
        self.assertIsInstance(r, declast.CXXClass)
        self.assertEqual("class Class1;", todict.print_node(r))
        self.assertEqual(todict.to_dict(r), {"name": "Class1"})

    def test_class2(self):
        """Forward declare class in a library"""
        library = ast.LibraryNode()
        library.add_declaration("class Class1;")
        self.assertIn("Class1", library.symbols)
        self.assertIsInstance(library.symbols["Class1"], ast.ClassNode)

        typedef = typemap.lookup_type("Class1")
        self.assertIsNotNone(typedef)
        self.assertEqual("Class1", typedef.name)
        self.assertEqual("Class1", typedef.cxx_type)
        self.assertEqual("shadow", typedef.base)

    def test_class2_node(self):
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
        library = ast.LibraryNode()
        ns = library.add_namespace("ns")
        ns.add_declaration("class Class2;")
        self.assertIn("Class2", ns.symbols)
        self.assertIsInstance(ns.symbols["Class2"], ast.ClassNode)

        typedef = typemap.lookup_type("ns::Class2")
        self.assertIsNotNone(typedef)
        self.assertEqual("ns::Class2", typedef.name)
        self.assertEqual("ns::Class2", typedef.cxx_type)


if __name__ == "__main__":
    unittest.main()
