# Copyright (c) 2017-2020, Lawrence Livermore National Security, LLC and
# other Shroud Project Developers.
# See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
########################################################################

from __future__ import print_function

from shroud import ast
from shroud import declast
from shroud import generate
from shroud import typemap  #

import unittest


class Namespace(unittest.TestCase):
    def test_ns1(self):
        lib = ast.LibraryNode()
        self.assertEqual("", lib.scope)

        # The Typemap must be created before the TypedefNode.
        ntypemap = typemap.Typemap("foo")
        typemap.register_type("foo", ntypemap)

        # typedef foo;
        # foo var;
        typefoo = lib.add_typedef("foo")
        self.assertIsInstance(typefoo, ast.TypedefNode)
        self.assertEqual("foo", typefoo.typemap.name)
        node = lib.qualified_lookup("foo")
        self.assertEqual(typefoo, node)
        self.assertEqual("foo", node.typemap.name)

        typ = lib.unqualified_lookup("foo")
        self.assertTrue(typ)

        std = lib.unqualified_lookup("std")
        self.assertIsNotNone(std)
        self.assertEqual("std::", std.scope)

        # Non existent names
        node = lib.unqualified_lookup("Nonexistent")
        self.assertIsNone(node)

    def test_ns2_class(self):
        # test class
        lib = ast.LibraryNode()
        class1 = lib.add_class("Class1")
        self.assertEqual("Class1", class1.typemap.name)
        self.assertEqual("Class1::", class1.scope)

        node = lib.qualified_lookup("Class1")
        self.assertEqual(class1, node)

        ns = lib.add_namespace("ns1")
        self.assertEqual("ns1::", ns.scope)

        class2 = ns.add_class("Class2")
        self.assertEqual("ns1::Class2", class2.typemap.name)
        self.assertEqual("ns1::Class2::", class2.scope)

        node = ns.unqualified_lookup("Class1")
        self.assertEqual(class1, node)
        node = ns.unqualified_lookup("Class2")
        self.assertEqual(class2, node)

        # look for Class2 in lib
        node = lib.unqualified_lookup("Class2")
        self.assertIsNone(node)

        # using namespace ns1
        lib.using_directive("ns1")
        node = lib.unqualified_lookup("Class2")
        self.assertEqual(class2, node)

    def test_ns3_enum(self):
        # test enum
        typemap.initialize()
        lib = ast.LibraryNode()
        enum1 = lib.add_enum("enum Enum1 {}")
        self.assertEqual("Enum1", enum1.typemap.name)
        self.assertEqual("Enum1::", enum1.scope)

        node = lib.qualified_lookup("Enum1")
        self.assertEqual(enum1, node)

        ns = lib.add_namespace("ns1")
        enum2 = ns.add_enum("enum Enum2 {}")
        self.assertEqual("ns1::Enum2", enum2.typemap.name)
        self.assertEqual("ns1::Enum2::", enum2.scope)

        node = ns.unqualified_lookup("Enum1")
        self.assertEqual(enum1, node)
        node = ns.unqualified_lookup("Enum2")
        self.assertEqual(enum2, node)

        # look for Enum2 in lib
        node = lib.unqualified_lookup("Enum2")
        self.assertIsNone(node)

        # using namespace ns1
        lib.using_directive("ns1")
        node = lib.unqualified_lookup("Enum2")
        self.assertEqual(enum2, node)

        # Add enum to class
        class1 = ns.add_class("Class1")
        enum3 = class1.add_enum("enum Enum3 {}")
        self.assertEqual("ns1::Class1::Enum3", enum3.typemap.name)
        self.assertEqual("ns1::Class1::Enum3::", enum3.scope)
        node = class1.qualified_lookup("Enum3")
        self.assertEqual(enum3, node)

    def test_ns4_namespace(self):
        # nested namespace
        lib = ast.LibraryNode()
        ns1 = lib.add_namespace("ns1")
        self.assertEqual("ns1::", ns1.scope)
        self.assertEqual(ns1, lib.qualified_lookup("ns1"))

        ns2 = ns1.add_namespace("ns2")
        self.assertEqual("ns1::ns2::", ns2.scope)
        self.assertEqual(ns2, ns1.qualified_lookup("ns2"))

        class1 = ns2.add_class("Class1")
        enumx = ns2.add_enum("enum Enumx {}")
        self.assertEqual(class1, ns2.qualified_lookup("Class1"))
        self.assertEqual(enumx, ns2.qualified_lookup("Enumx"))

        # from ns1, try to lookup Enumx
        node = ns1.unqualified_lookup("Enumx")
        self.assertIsNone(node)
        # 'using namespace ns2'
        self.assertEqual(0, len(ns1.using))
        node = ns1.using_directive("ns2")
        self.assertEqual(1, len(ns1.using))
        self.assertEqual(None, ns1.qualified_lookup("Enumx"))
        self.assertEqual(enumx, ns1.unqualified_lookup("Enumx"))

    def test_declare_namespace(self):
        lib = ast.LibraryNode("")
        ns = lib.add_declaration("namespace ns")


class CheckAst(unittest.TestCase):
    #    maxDiff = None
    def setUp(self):
        typemap.initialize()

    def test_a_library1(self):
        """Test LibraryNode"""
        library = ast.LibraryNode()

        self.assertEqual(library.language, "c++")
        self.assertEqual(library.options.wrap_c, True)
        self.assertEqual(library.options.wrap_fortran, True)

        fmt = library.fmtdict
        self.assertEqual(fmt.C_prefix, "DEF_")

    def test_a_library1(self):
        """Update LibraryNode"""
        library = ast.LibraryNode(
            language="c",
            options=dict(wrap_c=False),
            format=dict(C_prefix="XXX_", fmt1="fmt1value", fmt2="fmt2value"),
        )

        self.assertEqual(library.language, "c")  # updated from dict
        self.assertEqual(library.options.wrap_c, False)  # updated from dict
        self.assertEqual(library.options.wrap_fortran, True)
        self.assertEqual(library.fmtdict.fmt1, "fmt1value")
        self.assertEqual(library.fmtdict.fmt2, "fmt2value")

        fmt = library.fmtdict
        self.assertEqual(fmt.C_prefix, "XXX_")

    def test_b_function1(self):
        """Add a function to library"""
        library = ast.LibraryNode()
        library.add_function("void func1()")

        self.assertEqual(len(library.functions), 1)

    def test_b_function2(self):
        """Test options with function"""
        # Simulate YAML
        node = dict(
            options={"testa": "a", "testb": "b"},
            format={"fmt1": "f1", "fmt2": "f2"},
            declarations=[
                {
                    "decl": "void func1()",
                    "options": {"testc": "c"},
                    "format": {"fmt3": "f3"},
                },
                {
                    "block": True,
                    "options": {"testb": "bb", "testd": "d", "teste": "e"},
                    #                    'format': {
                    #                        'fmt2': 'f22',
                    #                        'fmt4': 'f4',
                    #                        'fmt5': 'f5',
                    #                    },
                    "declarations": [
                        {"decl": "void func2()", "options": {"teste": "ee"}}
                    ],
                },
            ],
        )
        library = ast.create_library_from_dictionary(node)

        self.assertEqual(len(library.functions), 2)
        self.assertEqual(library.options.testa, "a")
        self.assertEqual(library.options.testb, "b")
        self.assertEqual(library.fmtdict.fmt1, "f1")
        self.assertEqual(library.fmtdict.fmt2, "f2")

        self.assertEqual(library.functions[0].options.testa, "a")
        self.assertEqual(library.functions[0].options.testb, "b")
        self.assertEqual(library.functions[0].options.testc, "c")
        self.assertEqual(library.functions[0].fmtdict.fmt1, "f1")
        self.assertEqual(library.functions[0].fmtdict.fmt2, "f2")
        self.assertEqual(library.functions[0].fmtdict.fmt3, "f3")

        self.assertEqual(library.functions[1].options.testa, "a")
        self.assertEqual(library.functions[1].options.testb, "bb")
        self.assertNotIn("c", library.functions[1].options)
        self.assertEqual(library.functions[1].options.testd, "d")
        self.assertEqual(library.functions[1].options.teste, "ee")

    def test_c_class1(self):
        """Add a class to library"""
        library = ast.LibraryNode(format=dict(fmt1="f1", fmt2="f2"))
        library.add_class("Class1", format=dict(fmt2="f2", fmt3="f3"))

        self.assertEqual(library.fmtdict.fmt1, "f1")
        self.assertEqual(library.fmtdict.fmt2, "f2")
        self.assertEqual(len(library.classes), 1)

        self.assertEqual(library.classes[0].fmtdict.fmt1, "f1")
        self.assertEqual(library.classes[0].fmtdict.fmt2, "f2")
        self.assertEqual(library.classes[0].fmtdict.fmt3, "f3")

    def test_c_class2(self):
        """Add a classes with functions to library"""
        library = ast.LibraryNode()

        cls1 = library.add_class("Class1")
        cls1.add_function("void c1func1()")
        cls1.add_function("void c1func2()")

        cls2 = library.add_class("Class2")
        cls2.add_function("void c2func1()")

        self.assertEqual(len(library.classes), 2)
        self.assertEqual(len(library.classes[0].functions), 2)
        self.assertEqual(library.classes[0].functions[0].ast.name, "c1func1")
        self.assertEqual(library.classes[0].functions[1].ast.name, "c1func2")
        self.assertEqual(len(library.classes[1].functions), 1)
        self.assertEqual(library.classes[1].functions[0].ast.name, "c2func1")

    def test_c_class3(self):
        """Test class options"""
        # Simulate YAML
        node = dict(
            options={"testa": "a", "testb": "b", "testc": "c"},
            declarations=[
                {
                    "decl": "class Class1",
                    "options": {"testb": "bb"},
                    "declarations": [
                        {"decl": "void c1func1()", "options": {"testc": "cc"}},
                        {"decl": "void c1func2()"},
                    ],
                }
            ],
        )
        library = ast.create_library_from_dictionary(node)

        self.assertEqual(len(library.classes), 1)
        self.assertEqual(len(library.classes[0].functions), 2)

        self.assertEqual(library.options.testa, "a")
        self.assertEqual(library.options.testb, "b")
        self.assertEqual(library.options.testc, "c")

        self.assertEqual(library.classes[0].functions[0].options.testa, "a")
        self.assertEqual(library.classes[0].functions[0].options.testb, "bb")
        self.assertEqual(library.classes[0].functions[0].options.testc, "cc")

        self.assertEqual(library.classes[0].functions[1].options.testa, "a")
        self.assertEqual(library.classes[0].functions[1].options.testb, "bb")
        self.assertEqual(library.classes[0].functions[1].options.testc, "c")

    def test_class_template1(self):
        """Test class templates.
        """
        library = ast.LibraryNode()
        cls1 = library.add_class(
            "vector",
            template_parameters=["T"],
            cxx_template=[
                ast.TemplateArgument("<int>"),
                ast.TemplateArgument("<double>"),
            ],
        )
        self.assertIsInstance(cls1, ast.ClassNode)
        f1 = cls1.add_function("void push_back( const T& value );")
        self.assertIsInstance(f1, ast.FunctionNode)
        f2 = cls1.add_function("vector<T>()")
        self.assertIsInstance(f2, ast.FunctionNode)
        f3 = cls1.add_function("~vector<T>()")
        self.assertIsInstance(f2, ast.FunctionNode)

    def test_class_template2(self):
        """Test class templates.
        """
        library = ast.LibraryNode()
        cls1 = library.add_declaration(
            "template<typename T> class vector",
            cxx_template=[
                ast.TemplateArgument("<int>"),
                ast.TemplateArgument("<double>"),
            ],
        )
        self.assertIsInstance(cls1, ast.ClassNode)
        f1 = cls1.add_declaration("void push_back( const T& value );")
        self.assertIsInstance(f1, ast.FunctionNode)

    def test_d_generate1(self):
        """char bufferify
        Generate an additional function with len and len_trim attributes.
        """
        library = ast.LibraryNode()
        self.assertEqual(len(library.functions), 0)
        library.add_function("void func1(char * arg)")
        self.assertEqual(len(library.functions), 1)

        generate.generate_functions(library, None)
#        import json
#        from shroud import todict
#        print(json.dumps(todict.to_dict(library),
#                         indent=4, sort_keys=True, separators=(',', ': ')))
        
        self.assertEqual(len(library.functions), 2)
        self.assertEqual(
            library.functions[0].declgen,
            "void func1(char * arg +intent(inout))",
        )
        self.assertEqual(
            library.functions[1].declgen,
            "void func1(char * arg +intent(inout)+len+len_trim)",
        )

    def test_function_template1(self):
        """Test function templates.
        """
        library = ast.LibraryNode()
        fcn1 = library.add_function(
            "template<typename T> void func1(T arg)",
            cxx_template=[
                ast.TemplateArgument("<int>"),
                ast.TemplateArgument("<double>"),
            ],
        )

    def test_function_template2(self):
        """Test function templates.
        """
        library = ast.LibraryNode()
        cls1 = library.add_declaration(
            "template<typename T> void func1(T arg)",
            cxx_template=[
                ast.TemplateArgument("<int>"),
                ast.TemplateArgument("<double>"),
            ],
        )

    def test_e_enum1(self):
        """Add an enum to a library"""
        library = ast.LibraryNode()
        self.assertEqual(len(library.enums), 0)
        library.add_enum("enum Color{RED=1,BLUE,WHITE}")
        self.assertEqual(len(library.enums), 1)

        # parse functions which use the enum
        library.add_function("Color directionFunc(Color arg);")

    def test_e_enum2(self):
        """Add an enum to a namespace"""
        library = ast.LibraryNode()
        ns = library.add_namespace("ns")
        self.assertEqual(len(library.enums), 0)

        ns.add_enum("enum Color{RED=1,BLUE,WHITE}")
        self.assertEqual(len(ns.enums), 1)

        # parse global function which use the enum
        library.add_function("ns::Color directionFunc(ns::Color arg);")

    def test_e_enum3(self):
        """Add an enum to a class"""
        library = ast.LibraryNode()
        cls = library.add_class("Class1")
        self.assertEqual(len(cls.enums), 0)
        cls.add_enum("enum DIRECTION { UP = 2, DOWN, LEFT= 100, RIGHT };")
        self.assertEqual(len(cls.enums), 1)
        cls.add_function("DIRECTION directionFunc(DIRECTION arg);")

        # parse functions which use the enum
        library.add_function(
            "Class1::DIRECTION directionFunc(Class1::DIRECTION arg);"
        )

    def test_e_enum4(self):
        """enum errors"""
        library = ast.LibraryNode()
        with self.assertRaises(RuntimeError) as context:
            library.add_enum("void func1()")
        self.assertTrue(
            "Declaration is not an enumeration" in str(context.exception)
        )

        cls = library.add_class("Class1")
        with self.assertRaises(RuntimeError) as context:
            cls.add_enum("void func()")
        self.assertTrue(
            "Declaration is not an enumeration" in str(context.exception)
        )

if __name__ == "__main__":
    # Run a single test.
    suite = unittest.TestSuite()
    suite.addTest(CheckAst("test_d_generate1"))
    runner = unittest.TextTestRunner()
    runner.run(suite)
        
