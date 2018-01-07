from __future__ import print_function

from shroud import ast

import unittest

class CheckAst(unittest.TestCase):
#    maxDiff = None

    def test_a_library1(self):
        """Test LibraryNode"""
        library = ast.LibraryNode()

        self.assertEqual(library.language, 'c++')
        self.assertEqual(library.options.wrap_c, True)
        self.assertEqual(library.options.wrap_fortran, True)

        fmt = library._fmt
        self.assertEqual(fmt.C_prefix, 'DEF_')


    def test_a_library1(self):
        """Update LibraryNode"""
        node = dict(
            language='c',
            options=dict(
                wrap_c=False,
                C_prefix='XXX_',
            )
        )
        library = ast.LibraryNode(node)

        self.assertEqual(library.language, 'c')              # updated from dict
        self.assertEqual(library.options.wrap_c, False)      # updated from dict
        self.assertEqual(library.options.wrap_fortran, True)

        fmt = library._fmt
        self.assertEqual(fmt.C_prefix, 'XXX_')

    def test_b_function1(self):
        """Add a function to library"""
        node = dict(
            functions=[ dict(decl='void func1()') ]
        )
        library = ast.LibraryNode(node)

        self.assertEqual(len(library.functions), 1)

    def test_b_function2(self):
        """Test options with function"""
        node = dict(
            options={
                'testa': 'a',
                'testb': 'b',
            },
            functions=[
                {
                    'decl': 'void func1()',
                    'options': {
                        'testc': 'c',
                    }
                },{
                    'options': {
                        'testb': 'bb',
                        'testd': 'd',
                        'teste': 'e',
                    },
                },{
                    'decl': 'void func2()',
                    'options': {
                        'teste': 'ee',
                    }
                },
            ],
        )
        library = ast.LibraryNode(node)

        self.assertEqual(len(library.functions), 2)
        self.assertEqual(library.options.testa, 'a')
        self.assertEqual(library.options.testb, 'b')

        self.assertEqual(library.functions[0].options.testa, 'a')
        self.assertEqual(library.functions[0].options.testb, 'b')
        self.assertEqual(library.functions[0].options.testc, 'c')

        self.assertEqual(library.functions[1].options.testa, 'a')
        self.assertEqual(library.functions[1].options.testb, 'bb')
        self.assertNotIn('c', library.functions[1].options)
        self.assertEqual(library.functions[1].options.testd, 'd')
        self.assertEqual(library.functions[1].options.teste, 'ee')

    def test_c_class1(self):
        """Add a class to library"""
        node = dict(
            classes=[
                {
                    'name': 'Class1',
                }
            ]
        )
        library = ast.LibraryNode(node)

        self.assertEqual(len(library.classes), 1)

    def test_c_class2(self):
        """Add a class to library"""
        node = dict(
            classes=[
                {
                    'name': 'Class1',
                    'methods': [
                        {
                            'decl': 'void c1func1()',
                        },{
                            'decl': 'void c1func2()',
                        }
                    ],
                },{
                    'name': 'Class2',
                    'methods': [
                        {
                            'decl': 'void c2func1()',
                        }
                    ],
                },
            ],
        )
        library = ast.LibraryNode(node)

        self.assertEqual(len(library.classes), 2)
        self.assertEqual(len(library.classes[0].functions), 2)
        self.assertEqual(len(library.classes[1].functions), 1)


    def test_c_class2(self):
        """Test class options"""
        node = dict(
            options={
                'testa': 'a',
                'testb': 'b',
                'testc': 'c',
            },
            classes=[
                {
                    'name': 'Class1',
                    'options': {
                        'testb': 'bb',
                    },
                    'methods': [
                        {
                            'decl': 'void c1func1()',
                            'options': {
                                'testc': 'cc',
                            },
                        },{
                            'decl': 'void c1func2()',
                        }
                    ],
                },
            ],
        )
        library = ast.LibraryNode(node)

        self.assertEqual(len(library.classes), 1)
        self.assertEqual(len(library.classes[0].functions), 2)

        self.assertEqual(library.options.testa, 'a')
        self.assertEqual(library.options.testb, 'b')
        self.assertEqual(library.options.testc, 'c')

        self.assertEqual(library.classes[0].functions[0].options.testa, 'a')
        self.assertEqual(library.classes[0].functions[0].options.testb, 'bb')
        self.assertEqual(library.classes[0].functions[0].options.testc, 'cc')

        self.assertEqual(library.classes[0].functions[1].options.testa, 'a')
        self.assertEqual(library.classes[0].functions[1].options.testb, 'bb')
        self.assertEqual(library.classes[0].functions[1].options.testc, 'c')
