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

add_var_getter_setter will set meta attributes for the 
functions that it generates.

This file sets some meta attributes based on language specific
wrappings.

Fortran:
   intent
   api
   deref
   hidden
   owner    attrs


"""

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
            self.meta_function(cls, node)

    def set_func_intent(self, node, meta):
        declarator = node.ast.declarator
        if meta["intent"]:
            # getter/setter
            pass
        elif declarator.is_ctor():
            meta["intent"] = "ctor"
        elif declarator.is_dtor():
            meta["intent"] = "dtor"
        else:
            meta["intent"] = declarator.get_subprogram()

    def check_intent(self, arg):
        intent = arg.declarator.attrs["intent"]
        if intent:
            intent = intent.lower()
            if intent not in ["in", "out", "inout"]:
                intent = arg.declarator.attrs["intent"]
                self.cursor.generate("Bad value for intent: " + intent)
                intent = "inout"
            elif intent != "in" and not arg.declarator.is_indirect():
                # Nonpointers can only be intent(in).
                self.cursor.generate("Only pointer arguments may have intent of 'out' or 'inout'")
        return intent

    def check_value(self, arg):
        attrs = arg.declarator.attrs
        if attrs["value"] is None:
            if arg.declarator.is_indirect():
                if arg.typemap.name == "void":
                    # This causes Fortran to dereference the C_PTR
                    # Otherwise a void * argument becomes void **
                    if len(arg.declarator.pointer) == 1:
                        attrs["value"] = True  # void *
#                    else:
#                        attrs["value"] = None # void **  XXX intent(out)?
            else:
                attrs["value"] = True
        
    def set_arg_intent(self, node, arg, meta):
        """Set default intent meta-attribute.

        Intent is only valid on arguments.
        intent: lower case, no parens, must be in, out, or inout
        """
        if meta["intent"]:
            return
        declarator = arg.declarator
        intent = self.check_intent(arg)

        if intent is None:
            if declarator.is_function_pointer():
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

    def set_func_deref_c(self, node, meta):
        """
        XXX check meta attribute values?
        possible values: malloc
        """
        if meta["deref"]:
            return

    def set_func_deref_fortran(self, node, meta):
        """
        Function which return pointers or objects (std::string)
        set the deref meta attribute.
        """
        if meta["deref"]:
            return
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

    def set_arg_deref_c(self, arg, meta):
        """Check deref attr and set default for variable.

        Pointer variables set the default deref meta attribute.

        Use meta attributes define in YAML as:
          bind:
            c:
              decl: (arg+deref(malloc))
        """
        declarator = arg.declarator
        attrs = declarator.attrs
        ntypemap = arg.typemap
        is_ptr = declarator.is_indirect()

        deref = meta["deref"]
        if deref is not None:
            if deref not in ["malloc", "copy", "raw", "scalar"]:
                self.cursor.generate(
                    "Illegal value '{}' for deref attribute. "
                    "Must be 'malloc', 'copy', 'raw', "
                    "or 'scalar'.".format(deref)
                )
                return
            nindirect = declarator.is_indirect()
            if ntypemap.sgroup == "vector":
                pass
            elif nindirect != 2:
                self.cursor.generate(
                    "Can only have attribute 'deref' on arguments which"
                    " return a pointer:"
                    " '{}'".format(declarator.name))
            elif meta["intent"] == "in":
                self.cursor.generate(
                    "Cannot have attribute 'deref' on intent(in) argument"
                    " '{}'".format(declarator.name))
            return

        # Set deref attribute for arguments which return values.
        intent = meta["intent"]
        spointer = declarator.get_indirect_stmt()
        if ntypemap.name == "void":
            # void cannot be dereferenced.
            pass
        elif intent not in ["out", "inout"]:
            pass
        elif ntypemap.sgroup == "vector":
            meta["deref"] = "copy"
#        elif spointer in ["**", "*&"]:
#            if ntypemap.sgroup == "string":
#                # strings are not contiguous, so copy into argument.
#                meta["deref"] = "copy"
#            else:
#                meta["deref"] = "copy"

    def set_arg_deref_fortran(self, arg, meta):
        """Check deref attr and set default for variable.

        Pointer variables set the default deref meta attribute.
        """
        if meta["deref"]:
            return
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
#            if ntypemap.name == "void":
#                # void cannot be dereferenced.
            if ntypemap.sgroup == "vector":
                pass
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
        elif intent not in ["out", "inout"]:
            pass
        elif spointer in ["**", "*&"]:
            if ntypemap.sgroup == "string":
                # strings are not contiguous, so copy into argument.
                meta["deref"] = "copy"
            else:
                meta["deref"] = "pointer"

    def set_func_api(self, wlang, node, meta):
        """
        Based on other meta attrs: 
        """
        # from check_fcn_attrs  (C and Fortran)
        ast = node.ast
        ntypemap = ast.typemap
        api = ast.declarator.attrs["api"]
        shared = ast.declarator.metaattrs

        if api:
            # XXX - from check_common_attrs
            meta["api"] = api
        elif ntypemap.sgroup == "shadow":
            if node.return_this:
                meta["api"] = "this"
            elif node.options.C_shadow_result:
                meta["api"] = "capptr"
            else:
                meta["api"] = "capsule"

        if wlang == "c":
            return
        if meta["api"]:
            return
        if meta["deref"] == "raw":
            # No bufferify required for raw pointer result.
            return

        # arg_to_buffer
        fmt_func = node.fmtdict

        result_is_ptr = ast.declarator.is_indirect()
        # when the result is added as an argument to the Fortran api.

        # Check if result needs to be an argument.

        if node.options.F_CFI:
            result_as_arg = ""  # Only applies to string functions
            cfi_result = False
            if ntypemap.sgroup == "string":
                cfi_result   = "cfi"
                result_as_arg = fmt_func.F_string_result_as_arg
            elif ntypemap.sgroup == "char" and result_is_ptr:
                cfi_result   = "cfi"
                result_as_arg = fmt_func.F_string_result_as_arg
            elif meta["deref"] in ["allocatable", "pointer"]:
                cfi_result   = "cfi"
            if cfi_result:
                if result_as_arg:
                    meta["deref"] = "arg"
                meta["api"] = "cfi"
                return
        
        result_as_arg = ""  # Only applies to string functions
        need_buf_result = None
        if ntypemap.sgroup == "string":
            if meta["deref"] in ["allocatable", "pointer", "scalar"]:
                need_buf_result = "cdesc"
            else:
                need_buf_result = "buf"
            result_as_arg = fmt_func.F_string_result_as_arg
        elif ntypemap.sgroup == "char" and result_is_ptr:
            if meta["deref"] in ["allocatable", "pointer"]:
                # Result default to "allocatable".
                need_buf_result = "cdesc"
            else:
                need_buf_result = "buf"
            result_as_arg = fmt_func.F_string_result_as_arg
        elif ntypemap.base == "vector":
            need_buf_result = "cdesc"
        elif result_is_ptr:
            if meta["deref"] in ["allocatable", "pointer"]:
                if shared["dimension"]:
                    # int *get_array() +deref(pointer)+dimension(10)
                    need_buf_result = "cdesc"
        if need_buf_result:
            meta["api"] = need_buf_result
        if result_as_arg:
            meta["deref"] = "arg"
            meta["api"] = "buf"

    def set_arg_api_c(self, node, arg, meta):
        declarator = arg.declarator
        ntypemap = arg.typemap
        attrs = declarator.attrs
        shared = declarator.metaattrs
        api = attrs["api"]

        if meta["api"]:
            # API explicitly set by user.
            return

        if ntypemap.sgroup == "vector":
            meta["api"] = "buf"
        
    def set_arg_api_fortran(self, node, arg, meta):
        """
        Based on other meta attrs: deref
        """
        declarator = arg.declarator
        ntypemap = arg.typemap
        attrs = declarator.attrs
        shared = declarator.metaattrs
        api = attrs["api"]

        if api is None:
            pass
        elif api not in ["capi", "buf", "cdesc", "cfi"]:
                self.cursor.generate(
                    "'api' attribute must be 'capi', 'buf', 'cdesc' or 'cfi'"
                )
                api = None
        else:
            meta["api"] = api
            return

        if node.options.F_CFI:
            cfi_arg = False
            if shared["assumed-rank"]:
                cfi_arg = True
            elif attrs["rank"]:
                cfi_arg = True
            elif ntypemap.sgroup == "string":
                cfi_arg = True
            elif ntypemap.sgroup == "char":
                if declarator.is_indirect():
                    cfi_arg = True
            elif meta["deref"] in ["allocatable", "pointer"]:
                cfi_arg = True
            if cfi_arg:
                meta["api"] = "cfi"
                return
        
        has_buf_arg = None
        if ntypemap.sgroup == "string":
            if meta["deref"] in ["allocatable", "pointer", "copy"]:
                has_buf_arg = "cdesc"
                # XXX - this is not tested
                # XXX - tested with string **arg+intent(out)+dimension(ndim)
            else:
                has_buf_arg = "buf"
        elif ntypemap.sgroup == "char":
            if meta["ftrim_char_in"]:
                pass
            elif declarator.is_indirect():
                if meta["deref"] in ["allocatable", "pointer"]:
                    has_buf_arg = "cdesc"
                else:
                    has_buf_arg = "buf"
        elif ntypemap.sgroup == "vector":
            if meta["intent"] == "in":
                # Pass SIZE.
                has_buf_arg = "buf"
            else:
                has_buf_arg = "cdesc"
        elif (ntypemap.sgroup == "native" and
              meta["intent"] == "out" and
              meta["deref"] != "raw" and
              declarator.get_indirect_stmt() in ["**", "*&"]):
            # double **values +intent(out) +deref(pointer)
            has_buf_arg = "cdesc"
            #has_buf_arg = "buf" # XXX - for scalar?
        if has_buf_arg:
            meta["api"] = has_buf_arg
        
    def set_arg_hidden(self, arg, meta):
        """
        Fortran only.
        """
        declarator = arg.declarator
        hidden = declarator.attrs["hidden"]

        if hidden:
            meta["hidden"] = hidden

    def set_func_share(self, node, meta):
        """Use shared meta attribute unless already set.

        May already be set for getter/setter.
        """
        share_meta = statements.get_func_metaattrs(node, "share")

        if not meta["intent"]:
            meta["intent"] = share_meta["intent"]

    def set_arg_share(self, node, arg, meta):
        """Use shared meta attribute unless already set."""
        share_meta = statements.get_arg_metaattrs(node, arg, "share")

        if not meta["intent"]:
            meta["intent"] = share_meta["intent"]
            
######################################################################
#

class FillMetaShare(FillMeta):
    def meta_function(self, cls, node):
        wlang = "share"
        cursor = self.cursor
        func_cursor = cursor.push_node(node)
        #####
        ast = node.ast
        declarator = ast.declarator

        r_bind = statements.fetch_func_bind(node, wlang)
        r_meta = r_bind.meta

        self.set_func_intent(node, r_meta)

        # --- Loop over function parameters
        for arg in ast.declarator.params:
            func_cursor.arg = arg
            declarator = arg.declarator
            arg_name = declarator.user_name

            a_bind = statements.fetch_arg_bind(node, arg, wlang)
            meta = a_bind.meta

            self.set_arg_intent(node, arg, meta)

            if declarator.is_function_pointer():
                for arg1 in declarator.params:
                    attrs = arg1.declarator.attrs
                    attrs["intent"] = self.check_intent(arg1)
                    self.check_value(arg1)
            
        # --- End loop over function parameters
        func_cursor.arg = None
        cursor.pop_node(node)

######################################################################
#

class FillMetaC(FillMeta):
    def meta_function(self, cls, node):
        if not node.wrap.c:
            return
        wlang = "c"
        cursor = self.cursor
        func_cursor = cursor.push_node(node)
        #####
        ast = node.ast
        declarator = ast.declarator

        r_bind = statements.fetch_func_bind(node, wlang)
        r_meta = r_bind.meta

        self.set_func_share(node, r_meta)
        self.set_func_deref_c(node, r_meta)
        self.set_func_api(wlang, node, r_meta)

        # --- Loop over function parameters
        for arg in ast.declarator.params:
            func_cursor.arg = arg
            declarator = arg.declarator
            arg_name = declarator.user_name

            a_bind = statements.fetch_arg_bind(node, arg, wlang)
            meta = a_bind.meta

            self.set_arg_share(node, arg, meta)
            self.set_arg_deref_c(arg, meta)
            self.set_arg_api_c(node, arg, meta)
        # --- End loop over function parameters
        func_cursor.arg = None

        # Lookup statements if there are no meta attribute errors
        if node.wrap.c:
            stmt0 = statements.lookup_c_function_stmt(node)
            result_stmt = statements.lookup_local_stmts([wlang], stmt0, node)
            r_bind.stmt = result_stmt
            if stmt0 is not result_stmt:
                r_bind.fstmts = wlang
            for arg in ast.declarator.params:
                arg_stmt = statements.lookup_c_arg_stmt(node, arg)
                a_bind = statements.get_arg_bind(node, arg, wlang)
                a_bind.stmt = arg_stmt
        
        cursor.pop_node(node)
            
######################################################################
#

class FillMetaFortran(FillMeta):
    def meta_function(self, cls, node):
        if not node.wrap.fortran:
            return
        wlang = "f"
        cursor = self.cursor
        func_cursor = cursor.push_node(node)
        #####
        ast = node.ast
        declarator = ast.declarator

        r_bind = statements.fetch_func_bind(node, wlang)
        r_meta = r_bind.meta

        self.set_func_share(node, r_meta)
        self.set_func_deref_fortran(node, r_meta)
        self.set_func_api(wlang, node, r_meta)
        
        # --- Loop over function parameters
        for arg in ast.declarator.params:
            func_cursor.arg = arg
            declarator = arg.declarator
            arg_name = declarator.user_name

            a_bind = statements.fetch_arg_bind(node, arg, wlang)
            meta = a_bind.meta

            self.set_arg_share(node, arg, meta)
            self.set_arg_fortran(node, arg, meta)
            self.set_arg_deref_fortran(arg, meta)
            self.set_arg_api_fortran(node, arg, meta)
            self.set_arg_hidden(arg, meta)
        # --- End loop over function parameters
        func_cursor.arg = None

        # Lookup statements if there are no meta attribute errors
        if node.wrap.fortran:
            stmt0 = statements.lookup_f_function_stmt(node)
            result_stmt = statements.lookup_local_stmts([wlang], stmt0, node)
            r_bind.stmt = result_stmt
            if stmt0 is not result_stmt:
                r_bind.fstmts = wlang
            for arg in ast.declarator.params:
                arg_stmt = statements.lookup_f_arg_stmt(node, arg)
                a_bind = statements.get_arg_bind(node, arg, wlang)
                a_bind.stmt = arg_stmt
        
        cursor.pop_node(node)

    def set_arg_fortran(self, node, arg, meta):
        """
        Deal with Fortran specific attributes.
        """
        options = node.options
        declarator = arg.declarator
        attrs = declarator.attrs
        is_ptr = declarator.is_indirect()

        char_ptr_in = (
            is_ptr == 1 and
            meta["intent"] == "in" and
            arg.typemap.name == "char")
            
        blanknull = attrs["blanknull"]
        if blanknull is not None:
            if not char_ptr_in:
                self.cursor.generate(
                    "blanknull attribute can only be "
                    "used on intent(in) 'char *'"
                )
        elif char_ptr_in:
            blanknull = options.F_blanknull
        if blanknull:
            meta["blanknull"] = blanknull

        if attrs["api"]:  # User set
            pass
        elif (
            options.F_CFI is False and
            char_ptr_in and
            blanknull is False
        ):
            # const char *arg
            # char *arg+intent(in)
            # Add terminating NULL in Fortran wrapper.
            # Avoid a C wrapper just to do the NULL terminate.
            meta["ftrim_char_in"] = options.F_trim_char_in
        
######################################################################
#

class FillMetaPython(FillMeta):
    def meta_function(self, cls, node):
        if not node.wrap.python:
            return
        wlang = "py"
        cursor = self.cursor
        func_cursor = cursor.push_node(node)
        #####
        ast = node.ast
        declarator = ast.declarator

        r_bind = statements.fetch_func_bind(node, wlang)
        r_meta = r_bind.meta

        self.set_func_share(node, r_meta)
        self.set_func_deref(node, r_meta)
        stmt0 = None

        if stmt0:
            result_stmt = statements.lookup_local_stmts([wlang], stmt0, node)
            r_bind.stmt = result_stmt
            if stmt0 is not result_stmt:
                r_bind.fstmts = wlang

        # --- Loop over function parameters
        for arg in ast.declarator.params:
            func_cursor.arg = arg
            declarator = arg.declarator
            arg_name = declarator.user_name

            a_bind = statements.fetch_arg_bind(node, arg, wlang)
            meta = a_bind.meta

            self.set_arg_share(node, arg, meta)
            self.set_arg_deref(arg, meta)
            arg_stmt = None
            a_bind.stmt = arg_stmt

        # --- End loop over function parameters
        func_cursor.arg = None
        cursor.pop_node(node)

    def filter_deref(self, deref):
        """
        Filter top level decl values of deref
        to only allow a subset of raw and scalar.
        Remove values which are intended for Fortran:
        allocatable, pointer.
        """
        if deref in ["raw", "scalar"]:
            return deref
        return None
        
    def set_func_deref(self, node, meta):
        ast = node.ast
        declarator = ast.declarator
        attrs = declarator.attrs
        deref = self.filter_deref(attrs["deref"])

        if deref:
            meta["deref"] = deref

    def set_arg_deref(self, arg, meta):
        declarator = arg.declarator
        attrs = declarator.attrs
        deref = self.filter_deref(attrs["deref"])

        if deref:
            meta["deref"] = deref
        
######################################################################
#

class FillMetaLua(FillMeta):
    def meta_function(self, cls, node):
        if not node.wrap.python:
            return
        wlang = "lua"
        cursor = self.cursor
        func_cursor = cursor.push_node(node)
        #####
        ast = node.ast
        declarator = ast.declarator

        r_bind = statements.fetch_func_bind(node, wlang)
        r_meta = r_bind.meta

        self.set_func_share(node, r_meta)
        stmt0 = None

        if stmt0:
            result_stmt = statements.lookup_local_stmts([wlang], stmt0, node)
            r_bind.stmt = result_stmt
            if stmt0 is not result_stmt:
                r_bind.fstmts = wlang

        # --- Loop over function parameters
        for arg in ast.declarator.params:
            func_cursor.arg = arg
            declarator = arg.declarator
            arg_name = declarator.user_name

            a_bind = statements.fetch_arg_bind(node, arg, wlang)
            meta = a_bind.meta

            self.set_arg_share(node, arg, meta)
            arg_stmt = None
            a_bind.stmt = arg_stmt

        # --- End loop over function parameters
        func_cursor.arg = None
        cursor.pop_node(node)

######################################################################
#

process_map=dict(
    share=FillMetaShare,
    c=FillMetaC,
    f=FillMetaFortran,
    py=FillMetaPython,
    lua=FillMetaLua,
)

def process_metaattrs(newlibrary, wlang):
    process_map[wlang](newlibrary).meta_library()
        
