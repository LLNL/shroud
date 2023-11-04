# Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and
# other Shroud Project Developers.
# See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

"""
Set the meta attributes for each wrapper.
Derived from user supplied attributes as well a
defaults based on typemap.

generate.VerifyAttrs will do some initial error checking and
preprocessing on user supplied attributes that may apply to all
wrappers. These are in declarator.metaattrs.

This file sets some meta attributes based on language specific
wrappings.

Fortran:
   deref
   hidden

"""

import collections

from . import error
from . import statements

class FillMeta(object):
    """Loop over Nodes and fill meta attributes.
    """
    def __init__(self, newlibrary):
        self.newlibrary = newlibrary
        self.language = newlibrary.language
        self.cursor = error.get_cursor()

    def meta_library(self):
        self.meta_namespace(self.newlibrary.wrap_namespace)

    def meta_namespace(self, node):
        cursor = self.cursor
        
        for cls in node.classes:
            cursor.push_phase("FillMeta class function")
            self.meta_functions(cls, cls.functions)
            cursor.pop_phase("FillMeta class function")

        cursor.push_phase("FillMeta function")
        self.meta_functions(None, node.functions)
        cursor.pop_phase("FillMeta function")

        for ns in node.namespaces:
            self.meta_namespace(ns)

    def meta_functions(self, cls, functions):
        for node in functions:
            if node.wrap.c:
                self.meta_function("c", cls, node)
            if node.wrap.fortran:
                self.meta_function("f", cls, node)

    def meta_function(self, wlang, cls, node):
        cursor = self.cursor
        func_cursor = cursor.push_node(node)
        #####
        ast = node.ast
        declarator = ast.declarator

        bind = node._bind.setdefault(wlang, {})
        bind_result = bind.setdefault("+result", statements.BindArg())
        r_meta = bind_result.meta = collections.defaultdict(lambda: None)

        self.set_func_intent(node, r_meta)

        # --- Loop over function parameters
        for arg in ast.declarator.params:
            func_cursor.arg = arg
            declarator = arg.declarator
            arg_name = declarator.user_name

            bind_arg = bind.setdefault(arg_name, statements.BindArg())
            meta = bind_arg.meta = collections.defaultdict(lambda: None)

            self.set_arg_intent(node, arg, meta)

            
        # --- End loop over function parameters
        func_cursor.arg = None

        #####
        cursor.pop_node(node)

    def set_func_intent(self, node, meta):
        declarator = node.ast.declarator
        if declarator.is_ctor():
            meta["intent"] = "ctor"
        elif declarator.is_dtor():
            meta["intent"] = "dtor"
        else:
            meta["intent"] = declarator.get_subprogram()

    def set_arg_intent(self, node, arg, meta):
        declarator = arg.declarator
        intent = declarator.attrs["intent"]
        if intent is None:
            if node is None:
                # do not default intent for function pointers
                pass
            elif declarator.is_function_pointer():
                intent = "in"
            elif not declarator.is_indirect():
                intent = "in"
            elif arg.const:
                intent = "in"
            elif arg.typemap.sgroup == "void":
                # void *
                intent = "in"  # XXX must coordinate with VALUE
            else:
                intent = "inout"
            # XXX - Do hidden arguments need intent?
        else:
            intent = intent.lower()
        meta["intent"] = intent
