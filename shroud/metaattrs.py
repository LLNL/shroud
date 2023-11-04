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
   intent
   api
   deref
   hidden
   owner    attrs


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
        self.set_func_deref(node, r_meta)
        self.set_func_api(node, r_meta)

        # --- Loop over function parameters
        for arg in ast.declarator.params:
            func_cursor.arg = arg
            declarator = arg.declarator
            arg_name = declarator.user_name

            bind_arg = bind.setdefault(arg_name, statements.BindArg())
            meta = bind_arg.meta = collections.defaultdict(lambda: None)

            self.set_arg_intent(node, arg, meta)
            self.set_arg_deref(arg, meta)
            self.set_arg_api(arg, meta)

            
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

    def set_func_deref(self, node, meta):
        """
        Function which return pointers or objects (std::string)
        set the deref meta attribute.
        """
        # check_deref_attr_func
        ast = node.ast
        declarator = ast.declarator
        attrs = declarator.attrs
        deref = attrs["deref"]
        mderef = None
        ntypemap = ast.typemap
        nindirect = declarator.is_indirect()

        if declarator.get_subprogram() == "subroutine":
            pass
        if ntypemap.sgroup == "void":
            # Unable to set Fortran pointer for void
            # if deref set, error
            pass
        elif ntypemap.sgroup == "shadow":
            # Change a C++ pointer into a Fortran pointer
            # return 'void *' as 'type(C_PTR)'
            # 'shadow' assigns pointer to type(C_PTR) in a derived type
            # Array of shadow?
            pass
        elif ntypemap.sgroup == "string":
            if deref:
                mderef = deref
            elif attrs["len"]:
                mderef = "copy"
            else:
                mderef = "allocatable"
        elif ntypemap.sgroup == "vector":
            if deref:
                mderef = deref
            else:
                mderef = "allocatable"
        elif nindirect > 1:
            if deref:
                self.cursor.generate(
                    "Cannot have attribute 'deref' on function which returns multiple indirections")
        elif nindirect == 1:
            # pointer to a POD  e.g. int *
            if deref:
                mderef = deref
            elif ntypemap.sgroup == "char":  # char *
                if attrs["len"]:
                    mderef = "copy"
                else:
                    mderef = "allocatable"
            elif attrs["dimension"]:  # XXX - or rank?
                mderef = "pointer"
            else:
                mderef = node.options.return_scalar_pointer
        elif deref:
            self.cursor.generate("Cannot have attribute 'deref' on non-pointer function")
        meta["deref"] = mderef

    def set_arg_deref(self, arg, meta):
        """Check deref attr and set default for variable.

        Pointer variables set the default deref meta attribute.
        """
        # check_deref_attr_var
        # XXX - error via FortranGeneric
        declarator = arg.declarator
        attrs = declarator.attrs
        ntypemap = arg.typemap
        is_ptr = declarator.is_indirect()

        deref = attrs["deref"]
        if deref is not None:
            if deref not in ["allocatable", "pointer", "raw", "scalar"]:
                self.cursor.generate(
                    "Illegal value '{}' for deref attribute. "
                    "Must be 'allocatable', 'pointer', 'raw', "
                    "or 'scalar'.".format(deref)
                )
                return
            nindirect = declarator.is_indirect()
            if ntypemap.sgroup == "vector":
                if deref:
                    mderef = deref
                else:
                    # Copy vector to new array.
                    mderef = "allocatable"
            elif nindirect != 2:
                self.cursor.generate(
                    "Can only have attribute 'deref' on arguments which"
                    " return a pointer:"
                    " '{}'".format(declarator.name))
            elif meta["intent"] == "in":
                self.cursor.generate(
                    "Cannot have attribute 'deref' on intent(in) argument"
                    " '{}'".format(declarator.name))
            meta["deref"] = attrs["deref"]
            return

        # Set deref attribute for arguments which return values.
        intent = meta["intent"]
        spointer = declarator.get_indirect_stmt()
        if ntypemap.name == "void":
            # void cannot be dereferenced.
            pass
        elif spointer in ["**", "*&"] and intent == "out":
            if ntypemap.sgroup == "string":
                # strings are not contiguous, so copy into argument.
                meta["deref"] = "copy"
            else:
                meta["deref"] = "pointer"
    
    def set_func_api(self, node, meta):
        """
        Based on other meta attrs: 
        """
        # from check_fcn_attrs  (C and Fortran)
        ast = node.ast
        api = ast.declarator.attrs["api"]

        if api:
            # XXX - from check_common_attrs
            meta["api"] = api
        elif ast.typemap.sgroup == "shadow":
            if node.return_this:
                meta["api"] = "this"
            elif node.options.C_shadow_result:
                meta["api"] = "capptr"
            else:
                meta["api"] = "capsule"

    def set_arg_api(self, arg, meta):
        """
        Based on other meta attrs: 
        """
        declarator = arg.declarator
        api = declarator.attrs["api"]

        # XXX - from check_common_attrs
        if api:
            meta["api"] = api
        
        
