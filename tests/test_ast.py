# Copyright (c) 2017-2018, Lawrence Livermore National Security, LLC.
# Produced at the Lawrence Livermore National Laboratory
# 
# LLNL-CODE-738041.
# All rights reserved.
#  
# This file is part of Shroud.  For details, see
# https://github.com/LLNL/shroud. Please also read shroud/LICENSE.
#  
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#  
# * Redistributions of source code must retain the above copyright
#   notice, this list of conditions and the disclaimer below.
# 
# * Redistributions in binary form must reproduce the above copyright
#   notice, this list of conditions and the disclaimer (as noted below)
#   in the documentation and/or other materials provided with the
#   distribution.
# 
# * Neither the name of the LLNS/LLNL nor the names of its contributors
#   may be used to endorse or promote products derived from this
#   software without specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL LAWRENCE
# LIVERMORE NATIONAL SECURITY, LLC, THE U.S. DEPARTMENT OF ENERGY OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# 
########################################################################

from __future__ import print_function

from shroud import ast
from shroud import declast
from shroud import generate

import unittest

class CheckAst(unittest.TestCase):
#    maxDiff = None

    def test_a_library1(self):
        """Test LibraryNode"""
        library = ast.LibraryNode()

        self.assertEqual(library.language, 'c++')
        self.assertEqual(library.options.wrap_c, True)
        self.assertEqual(library.options.wrap_fortran, True)

        fmt = library.fmtdict
        self.assertEqual(fmt.C_prefix, 'DEF_')


    def test_a_library1(self):
        """Update LibraryNode"""
        library = ast.LibraryNode(
            language='c',
            options=dict(
                wrap_c=False,
            ),
            format=dict(
                C_prefix='XXX_',
                fmt1='fmt1value',
                fmt2='fmt2value',
            )
        )

        self.assertEqual(library.language, 'c')              # updated from dict
        self.assertEqual(library.options.wrap_c, False)      # updated from dict
        self.assertEqual(library.options.wrap_fortran, True)
        self.assertEqual(library.fmtdict.fmt1, 'fmt1value')
        self.assertEqual(library.fmtdict.fmt2, 'fmt2value')

        fmt = library.fmtdict
        self.assertEqual(fmt.C_prefix, 'XXX_')

    def test_b_function1(self):
        """Add a function to library"""
        library = ast.LibraryNode()
        library.add_function('void func1()')

        self.assertEqual(len(library.functions), 1)

    def test_b_function2(self):
        """Test options with function"""
        node = dict(
            options={
                'testa': 'a',
                'testb': 'b',
            },
            format={
                'fmt1': 'f1',
                'fmt2': 'f2',
            },
            functions=[
                {
                    'decl': 'void func1()',
                    'options': {
                        'testc': 'c',
                    },
                    'format': {
                        'fmt3': 'f3',
                    },
                },{
                    'options': {
                        'testb': 'bb',
                        'testd': 'd',
                        'teste': 'e',
                    },
#                    'format': {
#                        'fmt2': 'f22',
#                        'fmt4': 'f4',
#                        'fmt5': 'f5',
#                    },
                },{
                    'decl': 'void func2()',
                    'options': {
                        'teste': 'ee',
                    }
                },
            ],
        )
        library = ast.create_library_from_dictionary(node)

        self.assertEqual(len(library.functions), 2)
        self.assertEqual(library.options.testa, 'a')
        self.assertEqual(library.options.testb, 'b')
        self.assertEqual(library.fmtdict.fmt1, 'f1')
        self.assertEqual(library.fmtdict.fmt2, 'f2')

        self.assertEqual(library.functions[0].options.testa, 'a')
        self.assertEqual(library.functions[0].options.testb, 'b')
        self.assertEqual(library.functions[0].options.testc, 'c')
        self.assertEqual(library.functions[0].fmtdict.fmt1, 'f1')
        self.assertEqual(library.functions[0].fmtdict.fmt2, 'f2')
        self.assertEqual(library.functions[0].fmtdict.fmt3, 'f3')

        self.assertEqual(library.functions[1].options.testa, 'a')
        self.assertEqual(library.functions[1].options.testb, 'bb')
        self.assertNotIn('c', library.functions[1].options)
        self.assertEqual(library.functions[1].options.testd, 'd')
        self.assertEqual(library.functions[1].options.teste, 'ee')

    def test_c_class1(self):
        """Add a class to library"""
        library = ast.LibraryNode(
            format=dict(
                fmt1='f1',
                fmt2='f2')
        )
        library.add_class('Class1',
                          format=dict(
                              fmt2='f2',
                              fmt3='f3')
        )

        self.assertEqual(library.fmtdict.fmt1, 'f1')
        self.assertEqual(library.fmtdict.fmt2, 'f2')
        self.assertEqual(len(library.classes), 1)

        self.assertEqual(library.classes[0].fmtdict.fmt1, 'f1')
        self.assertEqual(library.classes[0].fmtdict.fmt2, 'f2')
        self.assertEqual(library.classes[0].fmtdict.fmt3, 'f3')

    def test_c_class2(self):
        """Add a classes with functions to library"""
        library = ast.LibraryNode()

        cls1 = library.add_class('Class1')
        cls1.add_function('void c1func1()')
        cls1.add_function('void c1func2()')

        cls2 = library.add_class('Class2')
        cls2.add_function('void c2func1()')

        self.assertEqual(len(library.classes), 2)
        self.assertEqual(len(library.classes[0].functions), 2)
        self.assertEqual(library.classes[0].functions[0]._ast.name, 'c1func1')
        self.assertEqual(library.classes[0].functions[1]._ast.name, 'c1func2')
        self.assertEqual(len(library.classes[1].functions), 1)
        self.assertEqual(library.classes[1].functions[0]._ast.name, 'c2func1')

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
        library = ast.create_library_from_dictionary(node)

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

    def test_d_generate1(self):
        """char bufferify
        Geneate an additional function with len and len_trim attributes.
        """
        library = ast.LibraryNode()
        self.assertEqual(len(library.functions), 0)
        library.add_function('void func1(char * arg)')
        self.assertEqual(len(library.functions), 1)

        generate.generate_functions(library, None)
        self.assertEqual(len(library.functions), 2)
        self.assertEqual(library.functions[0].declgen,
                         'void func1(char * arg +intent(inout))')
        self.assertEqual(library.functions[1].declgen,
                         'void func1(char * arg +intent(inout)+len(Narg)+len_trim(Larg))')

#        import json
#        from shroud import util
#        print(json.dumps(library, cls=util.ExpandedEncoder, indent=4, sort_keys=True))


    def test_e_enum1(self):
        """Add an enum to a library"""
        library = ast.LibraryNode()
        self.assertEqual(len(library.enums), 0)
        library.add_enum('enum Color{RED=1,BLUE,WHITE}')
        self.assertEqual(len(library.enums), 1)

        # parse functions which use the enum
        library.add_function('Color directionFunc(Color arg);')

    def test_e_enum2(self):
        """Add an enum to a namespace"""
        library = ast.LibraryNode()
        ns = library.add_namespace('ns')
        self.assertEqual(len(library.enums), 0)
        self.assertEqual('ns::', ns.fmtdict.ns_scope)

        ns.add_enum('enum Color{RED=1,BLUE,WHITE}')
        self.assertEqual(len(library.enums), 1)

        # parse global function which use the enum
#        library.add_function('ns::Color directionFunc(ns::Color arg);')

    def test_e_enum3(self):
        """Add an enum to a class"""
        library = ast.LibraryNode()
        cls = library.add_class('Class1')
        self.assertEqual(len(cls.enums), 0)
        cls.add_enum('enum Color{RED=1,BLUE,WHITE}')
        self.assertEqual(len(cls.enums), 1)

        # parse functions which use the enum
#        library.add_function('Class1::DIRECTION directionFunc(Class1::DIRECTION arg);')

