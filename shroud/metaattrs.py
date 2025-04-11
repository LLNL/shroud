# Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
# other Shroud Project Developers.
# See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

"""
Set the meta attributes for each wrapper.
Derived from user supplied attributes as well a
defaults based on typemap.
Wrappers should use the meta attributes instead of using
the parsed attributes directly.

generate.VerifyAttrs will do some initial error checking and
preprocessing on user supplied attributes that may apply to all
wrappers. These are in declarator.metaattrs.

add_var_getter_setter will set meta attributes for the 
functions that it generates.

This file sets some meta attributes based on language specific
wrappings.

Shared:
   abstract Computed from declaration  ex  'native', 'native*'
   charlen  attrs

Fortran:
   intent
   api
   deref
   hidden
   len      attrs
   owner    attrs
   value

   fptr - FunctionNode for callback
      Converted from a function pointer into a function.
   ftrim_char_in - option.F_trim_char_in


"""

import copy

from . import ast
from . import declast
from .declstr import gen_decl
from . import error
from . import statements

FunctionNode = ast.FunctionNode

# Unique, non-None default.
missing = object()

class FillMeta(object):
    """Loop over Nodes and fill meta attributes.
    """
    def __init__(self, newlibrary, wlang):
        self.newlibrary = newlibrary
        self.language = newlibrary.language
        self.wlang = wlang
        self.cursor = error.get_cursor()

    def meta_library(self):
        self.meta_namespace(self.newlibrary.wrap_namespace)

    def meta_namespace(self, node):
        cursor = self.cursor
        
        for cls in node.classes:
            cursor.push_phase("FillMeta class function")
            for var in cls.variables:
                cursor.push_node(var)
                self.meta_variable(cls, var)
                cursor.pop_node(var)
            for func in cls.functions:
                cursor.push_node(func)
                self.meta_function(cls, func)
                cursor.pop_node(func)
            cursor.pop_phase("FillMeta class function")

        cursor.push_phase("FillMeta typedef")
        for typ in node.typedefs:
            cursor.push_node(typ)
            self.meta_typedef(None, typ)
            cursor.pop_node(typ)
        cursor.pop_phase("FillMeta typedef")

        cursor.push_phase("FillMeta function")
        for func in node.functions:
            cursor.push_node(func)
            self.meta_function(None, func)
            cursor.pop_node(func)
        cursor.pop_phase("FillMeta function")

        for ns in node.namespaces:
            self.meta_namespace(ns)

    def check_var_attrs(self, node, meta):
        """Check attributes for variables.
        This includes struct and class members.

        Args:
            node -
            meta - 
        """
        ast = node.ast
        declarator = ast.declarator
        attrs = declarator.attrs
        for attr in attrs:
            # XXX - deref on class/struct members
            if attr not in ["name", "readonly", "dimension", "deref"]:
                self.cursor.generate(
                    "Illegal attribute '{}' for variable '{}'".format(
                        attr, node.name
                    ) + "\nonly 'name', 'readonly', 'dimension' and 'deref' are allowed on variables"
                )

        dim = attrs.get("dimension", missing)
        if dim is not missing:
            is_ptr = declarator.is_indirect()
            if not is_ptr:
                self.cursor.generate(
                    "dimension attribute can only be "
                    "used on pointer and references"
                )
            self.parse_dim_attrs(dim, meta)

    def set_func_abstract_type(self, node, meta):
        meta["abstract"] = statements.find_abstract_declarator(node.ast)

    def set_arg_abstract_type(self, node, arg, meta):
        meta["abstract"] = statements.find_abstract_declarator(arg)
        
    def set_func_intent(self, node, meta):
        declarator = node.ast.declarator
        intent = declarator.attrs.get("intent", missing)
        if intent is not missing:
            intent = intent.lower()
            if intent not in ["getter", "setter"]:
                self.cursor.generate("Bad value for function intent: {}"
                                     .format(intent))
            meta["intent"] = intent
        elif declarator.is_ctor:
            meta["intent"] = "ctor"
        elif declarator.is_dtor:
            meta["intent"] = "dtor"
        else:
            meta["intent"] = declarator.get_subprogram()

        operator = declarator.attrs.get("operator", missing)
        if operator is not missing:
            meta["operator"] = operator

        custom = declarator.attrs.get("custom", missing)
        if custom is not missing:
            meta["custom"] = custom

    def check_intent(self, arg):
        intent = arg.declarator.attrs.get("intent", missing)
        if intent is missing:
            intent = None
        else:
            intent = intent.lower()
            if intent in ["getter", "setter"]:
                pass
            elif intent not in ["in", "out", "inout"]:
                intent = arg.declarator.attrs["intent"]
                self.cursor.generate("Bad value for argument {} intent: {}"
                                     .format(arg.declarator.user_name, intent))
                intent = "inout"
            elif intent != "in" and not arg.declarator.is_indirect():
                # Nonpointers can only be intent(in).
                self.cursor.generate("Only pointer arguments may have intent of 'out' or 'inout'")
        return intent

    def check_value(self, arg, meta):
        value = arg.declarator.attrs.get("value", missing)
        if value is missing:
            attrs = arg.declarator.attrs
            if arg.declarator.is_indirect():
                if arg.typemap.name == "void":
                    # This causes Fortran to dereference the C_PTR
                    # Otherwise a void * argument becomes void **
                    if meta["assumedtype"]:
                        # assumed-type interoperates with a C argument declared as "void *"
                        pass
                    elif len(arg.declarator.pointer) == 1:
                        meta["value"] = True  # void *
#                    else:
#                        meta["value"] = None # void **  XXX intent(out)?
            elif arg.typemap.sgroup == "string":
                # Strings pass a char *.
                pass
            else:
                meta["value"] = True
        else:
            meta["value"] = value
        
    def set_arg_intent(self, arg, meta, is_fptr=False):
        """Set default intent meta-attribute.

        Intent is only valid on arguments.
        intent: lower case, no parens, must be in, out, or inout
        """
        if meta["intent"]:
            return
        declarator = arg.declarator
        intent = self.check_intent(arg)

        if intent is None:
            if is_fptr:
                # intent is not defaulted for function pointer arguments
                # for historical reasons.
                intent = "none"
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
        Also applies to getter.
        """
        if meta["deref"]:
            return
        # check_deref_attr_func
        options = node.options
        ast = node.ast
        declarator = ast.declarator
        attrs = declarator.attrs
        deref = attrs.get("deref", missing)
        mderef = None
        ntypemap = ast.typemap
        nindirect = declarator.is_indirect()

        if declarator.get_subprogram() == "subroutine":
            pass
        if ntypemap.sgroup == "void":
            # Unable to set Fortran pointer for void
            # if deref set, error
            pass
        elif ntypemap.sgroup in ["smartptr", "shadow"]:
            # Change a C++ pointer into a Fortran pointer
            # return 'void *' as 'type(C_PTR)'
            # 'shadow' assigns pointer to type(C_PTR) in a derived type
            # Array of shadow?
            pass
        elif ntypemap.sgroup == "struct":
            if deref is not missing:
                mderef = deref
            elif nindirect == 1:
                mderef = "pointer"
            elif nindirect > 1:
                mderef = "raw"
        elif ntypemap.sgroup == "char":
            if deref is not missing:
                mderef = deref
            elif nindirect == 1:
                if "len" in attrs:
                    mderef = "copy"
                else:
                    mderef = options.F_deref_func_character
            elif nindirect > 1:
                mderef = "raw"
        elif ntypemap.sgroup == "string":
            if deref is not missing:
                mderef = deref
            elif "len" in attrs:
                mderef = "copy"
            else:
                mderef = options.F_deref_func_character
        elif ntypemap.implied_array:
            if deref is not missing:
                mderef = deref
            else:
                mderef = options.F_deref_func_implied_array
        elif nindirect > 1:
            if deref is missing:
                deref = "raw"
            elif deref != "raw":
                self.cursor.generate(
                    "Multiple function indirections only supports 'deref(raw)', not '{}'".
                    format(deref))
        elif nindirect == 1:
            # pointer to a POD  e.g. int *
            if deref is not missing:
                mderef = deref
            elif "dimension" in attrs:  # XXX - or rank?
                mderef = options.F_deref_func_array
            else:
                mderef = options.return_scalar_pointer  # options.F_deref_func_scalar
        elif deref is not missing:
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
            if ntypemap.implied_array:
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
        if declarator.is_function_pointer() or ntypemap.base == "procedure":
            if attrs.get("external"):
                meta["deref"] = "external"
            elif attrs.get("funptr"):
                meta["deref"] = "funptr"
        elif ntypemap.name == "void":
            # void cannot be dereferenced.
            pass
        elif intent not in ["out", "inout"]:
            pass
        elif ntypemap.implied_array:
            meta["deref"] = "copy"
#        elif spointer in ["**", "*&"]:
#            if ntypemap.sgroup == "string":
#                # strings are not contiguous, so copy into argument.
#                meta["deref"] = "copy"
#            else:
#                meta["deref"] = "copy"

    def set_arg_deref_fortran(self, node, arg, meta):
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

        deref = attrs.get("deref", missing)
        if deref is not missing:
            if deref not in ["allocatable", "pointer", "copy", "raw", "scalar"]:
                self.cursor.generate(
                    "Illegal value '{}' for deref attribute. "
                    "Must be 'allocatable', 'pointer', 'copy', 'raw', "
                    "or 'scalar'.".format(deref)
                )
                return
            nindirect = declarator.is_indirect()
#            if ntypemap.name == "void":
#                # void cannot be dereferenced.
            if ntypemap.implied_array:
                pass
            elif nindirect > 2:
                if deref != "raw":
                    self.cursor.generate(
                        "Multiple argument indirections only supports 'deref(raw)', not '{}'".
                        format(deref))
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
        options = node.options
        if declarator.is_function_pointer() or ntypemap.base == "procedure":
            if attrs.get("external"):
                meta["deref"] = "external"
            elif attrs.get("funptr"):
                meta["deref"] = "funptr"
        elif ntypemap.name == "void":
            # void cannot be dereferenced.
            pass
        elif intent not in ["out", "inout"]:
            pass
        elif ntypemap.implied_array:
            meta["deref"] = options.F_deref_arg_implied_array
        elif declarator.is_indirect() > 2:
            meta["deref"] = "raw"
        elif spointer in ["**", "*&"]:
            if ntypemap.sgroup == "char":
                meta["deref"] = options.F_deref_arg_character
            elif ntypemap.sgroup == "string":
                meta["deref"] = options.F_deref_arg_character
            elif "dimension" in attrs:  # XXX - or rank?
                meta["deref"] = options.F_deref_arg_array
            else:
                meta["deref"] = options.F_deref_arg_scalar

    def set_func_api_c(self, node, meta):
        """
        Based on other meta attrs: 
        """
        ast = node.ast
        ntypemap = ast.typemap
        attrs = ast.declarator.attrs
        api = attrs.get("api", missing)

        if api is not missing:
            meta["api"] = api
        elif ntypemap.sgroup in ["smartptr", "shadow"]:
            if node.return_this:
                meta["api"] = "this"
            elif node.options.C_shadow_result:
                meta["api"] = "capptr"
            else:
                meta["api"] = "capsule"

    def set_func_api_fortran(self, node, meta):
        """
        Based on other meta attrs: 
        """
        ast = node.ast
        ntypemap = ast.typemap
        attrs = ast.declarator.attrs
        api = attrs.get("api", missing)

        if api is not missing:
            if api == "capptr":
                # capptr is not used with Fortran wrappers.
                api = "capsule"
            meta["api"] = api
        elif ntypemap.sgroup in ["smartptr", "shadow"]:
            if node.return_this:
                meta["api"] = "this"
            else:
                meta["api"] = "capsule"

        if meta["api"]:
            return
        if meta["deref"] == "raw" and not meta["dimension"]:
            # No bufferify required for raw pointer result.
            # Return a type(C_PTR).
            return

        # arg_to_buffer
        fmt_func = node.fmtdict

        is_ptr = ast.declarator.is_indirect()
        # when the result is added as an argument to the Fortran api.

        # Check if result needs to be an argument.

        if node.options.F_CFI:
            cfi_result = False
            if ntypemap.sgroup == "string":
                cfi_result   = "cfi"
            elif ntypemap.sgroup == "char" and is_ptr:
                cfi_result   = "cfi"
            elif meta["deref"] in ["allocatable", "pointer"]:
                cfi_result   = "cfi"
            if cfi_result:
                meta["api"] = "cfi"
                return
        
        need_buf_result = None
        if ntypemap.sgroup == "string":
            if meta["deref"] in ["allocatable", "pointer", "scalar"]:
                need_buf_result = "cdesc"
            else:
                need_buf_result = "buf"
        elif ntypemap.sgroup == "char" and is_ptr:
            if meta["deref"] in ["allocatable", "pointer"]:
                # Result default to "allocatable".
                need_buf_result = "cdesc"
            else:
                need_buf_result = "buf"
        elif ntypemap.base == "struct":
            if is_ptr:
                need_buf_result = "cdesc"
        elif ntypemap.implied_array:
            need_buf_result = "cdesc"
        elif is_ptr:
            if meta["deref"] in ["allocatable", "pointer"]:
                if meta["dimension"]:
                    # int *get_array() +deref(pointer)+dimension(10)
                    need_buf_result = "cdesc"
        if need_buf_result:
            meta["api"] = need_buf_result

    def set_func_post_c(self, cls, node, meta):
        """Final check on metaattributes for C.
        Check defaults from typemap.
        """
        ast = node.ast
        ntypemap = ast.typemap

        if cls and cls.C_shared_class:
            if ast.is_ctor:
                meta["owner"] = cls.typemap.smart_pointer
                meta["api"] = "capptr"
            elif ast.is_dtor:
                meta["owner"] = cls.typemap.smart_pointer

    def set_func_post_fortran(self, cls, node, meta):
        """Final check on metaattributes for Fortran.
        Check defaults from typemap.
        """
        ast = node.ast
        ntypemap = ast.typemap

        if cls and cls.C_shared_class:
            if ast.is_ctor:
                meta["owner"] = cls.typemap.smart_pointer
                meta["api"] = "capsule"
                meta["deref"] = None
            elif ast.is_dtor:
                meta["owner"] = cls.typemap.smart_pointer
            
    def set_arg_api_c(self, arg, meta):
        declarator = arg.declarator
        ntypemap = arg.typemap
        attrs = declarator.attrs
        api = attrs.get("api", missing)

        if api is not missing:
            # API explicitly set by user.
            return

        if ntypemap.implied_array:
            meta["api"] = "buf"
        
    def set_arg_api_fortran(self, node, arg, meta, fptr_arg):
        """
        Based on other meta attrs: deref

        fptr_arg : True if processing function pointer arguments.
        """
        declarator = arg.declarator
        ntypemap = arg.typemap
        attrs = declarator.attrs
        api = attrs.get("api", missing)

        if api is missing:
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
            if meta["dimension"] == "..":   # assumed-rank
                cfi_arg = True
            elif meta["rank"]:
                cfi_arg = True
            elif ntypemap.sgroup == "string":
                cfi_arg = True
            elif ntypemap.sgroup == "char":
                if meta["deref"] == "copy":
                    # Copying into a CHARACTER argument.
                    pass
                elif declarator.is_indirect():
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
            if fptr_arg:
                # Pass raw pointer for function pointer arguments.
                pass
            elif meta["ftrim_char_in"]:
                pass
            elif declarator.is_indirect():
                if meta["deref"] in ["allocatable", "pointer"]:
                    has_buf_arg = "cdesc"
                else:
                    has_buf_arg = "buf"
        elif ntypemap.implied_array:
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
        Fortran/Python only.
        """
        declarator = arg.declarator
        hidden = declarator.attrs.get("hidden", missing)

        if hidden is not missing:
            meta["hidden"] = hidden

    def set_func_share(self, node, meta):
        """Use shared meta attribute unless already set.

        May already be set for getter/setter.
        """
        share_meta = statements.get_func_bind(node, "share").meta

        if not meta["intent"]:
            meta["intent"] = share_meta["intent"]
        for attr in [
                "abstract", "assumedtype",
                "custom", "dimension", "dim_ast",
                "free_pattern", "funcarg", "hidden",
                "len", "operator", "owner", "rank",
        ]:
            meta[attr] = share_meta[attr]

    def set_arg_share(self, node, arg, meta):
        """Use shared meta attribute unless already set."""
        share_meta = statements.get_arg_bind(node, arg, "share").meta

        if not meta["intent"]:
            meta["intent"] = share_meta["intent"]
        for attr in [
                "abstract", "assumedtype",
                "dimension", "dim_ast",
                "fptr", "free_pattern", "hidden", "len", "owner", "rank",
                "value", "optional",
        ]:
            meta[attr] = share_meta[attr]
        
    def set_typedef_share(self, node, meta):
        """Use shared meta attribute unless already set.
        """
        share_meta = statements.fetch_typedef_bind(node, "share").meta
        for attr in [
                "fptr",
        ]:
            meta[attr] = share_meta[attr]

######################################################################
#

class FillMetaShare(FillMeta):
    def meta_typedef(self, cls, node):
        # node - ast.TypedefNode
        wlang = self.wlang
        func_cursor = self.cursor.current
        #####
        r_bind = statements.fetch_typedef_bind(node, wlang)

        arg = node.ast
        if arg.declarator.is_function_pointer():
            newarg = dereference_function_pointer(arg)
            fptr = FunctionNode(gen_decl(newarg), parent=node, ast=newarg)
            r_bind.meta["fptr"] = fptr
            self.meta_function(None, fptr, is_fptr=True)

    def meta_variable(self, cls, node):
        wlang = self.wlang
        node_cursor = self.cursor.current
        bind = statements.fetch_var_bind(node, wlang)

        self.check_var_attrs(node, bind.meta)

    def meta_function(self, cls, node, is_fptr=False):
        # node - ast.FunctionNode
        wlang = self.wlang
        func_cursor = self.cursor.current
        #####
        r_bind = statements.fetch_func_bind(node, wlang)
        r_meta = r_bind.meta
        
        self.check_func_attrs(node, r_meta)
        self.set_func_abstract_type(node, r_meta)
        self.set_func_intent(node, r_meta)
        self.meta_function_params(node, is_fptr)

    def meta_function_params(self, node, is_fptr=False):
        """Set function argument meta attributes.
        Also used with function pointers arguments.
        """
        wlang = self.wlang
        func_cursor = self.cursor.current
        # --- Loop over function parameters
        for arg in node.ast.declarator.params:
            func_cursor.arg = arg

            a_bind = statements.fetch_arg_bind(node, arg, wlang)
            meta = a_bind.meta

            self.check_arg_attrs(node, arg, meta)
            self.set_arg_abstract_type(node, arg, meta)
            self.set_arg_intent(arg, meta, is_fptr)
            self.check_value(arg, meta)

            if node.options.F_default_args == "optional" and arg.declarator.init is not None:
                meta["optional"] = True

            if arg.declarator.is_function_pointer():
                # Convert function pointer into function
                newarg = dereference_function_pointer(arg)
                fptr = FunctionNode(gen_decl(newarg), parent=node, ast=newarg)
                meta["fptr"] = fptr
                self.meta_function(None, fptr, is_fptr=True)
        # --- End loop over function parameters
        func_cursor.arg = None

    def check_func_attrs(self, node, meta):
        cursor = self.cursor

        ast = node.ast
        declarator = ast.declarator
        attrs = declarator.attrs
#        node._has_found_default = False

        for attr in attrs:
            if attr not in [
                "api",          # arguments to pass to C wrapper.
                "allocatable",  # return a Fortran ALLOCATABLE
                "custom",
                "deref",  # How to dereference pointer
                "dimension",
                "free_pattern",
                "funcarg",
                "intent",    # getter/setter
                "len",
                "name",
                "operator",
                "owner",
                "pure",
                "rank",

                "external", # Only on function pointer
                "funptr",   # Only on function pointer
                "__line__",
            ]:
                cursor.generate(
                    "Illegal attribute '{}' for function '{}'".format(
                        attr, node.name
                    )
                )

        self.check_common_attrs(node.ast, meta)

        funcarg = attrs.get("funcarg", missing)
        if funcarg is not missing:
            if funcarg is True:
                meta["funcarg"] = node.options.F_result_as_arg
            else:
                meta["funcarg"] = funcarg

    def check_arg_attrs(self, node, arg, meta):
        cursor = self.cursor
        declarator = arg.declarator
        argname = declarator.user_name
        attrs = declarator.attrs

        for attr in attrs:
            if attr not in [
                "api",
                "allocatable",
                "assumedtype",
                "blanknull",   # Treat blank string as NULL pointer.
                "charlen",   # Assumed length of intent(out) char *.
                "external",
                "deref",
                "dimension",
                "funptr",
                "hidden",  # omitted in Fortran API, returned from C++
                "implied",  # omitted in Fortran API, value passed to C++
                "intent",
                "len",
                "len_trim",
                "name",
                "owner",
                "pass",
                "rank",
                "size",
                "value",
                "__line__",
            ]:
                cursor.generate(
                    "Illegal attribute '{}' for argument '{}'".format(
                        attr, argname))
                continue

        arg_typemap = arg.typemap
        if arg_typemap is None:
            # Sanity check to make sure arg_typemap exists
            raise RuntimeError(
                "check_arg_attrs: Missing arg.typemap on line {}: {}".format(
                    node.linenumber, node.decl
                )
            )

        self.check_common_attrs(arg, meta)

        # assumedtype
        assumedtype = attrs.get("assumedtype", missing)
        if assumedtype is not missing:
            if "value" in attrs:
                cursor.generate(
                    "argument '{}' must not have value=True "
                    "because it has the assumedtype attribute.".format(argname)
                )
            meta["assumedtype"] = assumedtype

        charlen = attrs.get("charlen", missing)
        if charlen is not missing:
            meta["charlen"] = charlen
            
    def check_common_attrs(self, ast, meta):
        """Check attributes which are common to function and argument AST
        This includes: dimension, free_pattern, owner, rank

        Parameters
        ----------
        ast : declast.Declaration
        """
        declarator = ast.declarator
        attrs = declarator.attrs
        ntypemap = ast.typemap
        is_ptr = declarator.is_indirect()

        # dimension
        rank = attrs.get("rank", missing)
        if rank is not missing:
            if rank is True:
                self.cursor.generate(
                    "'rank' attribute must have an integer value"
                )
            else:
                try:
                    rank = int(rank)
                except ValueError:
                    self.cursor.generate(
                        "rank attribute must have an integer value, not '{}'"
                        .format(rank)
                    )
                else:
                    meta["rank"] = int(rank)
                    if rank > 7:
                        self.cursor.generate(
                            "'rank' attribute must be 0-7, not '{}'"
                            .format(rank)
                        )
            if not is_ptr:
                self.cursor.generate(
                    "rank attribute can only be "
                    "used on pointer and references"
                )

        dimension = attrs.get("dimension", missing)
        if dimension is not missing:
            if dimension is True:
                self.cursor.generate(
                    "dimension attribute must have a value."
                )
                dimension = None
            if "value" in attrs:
                self.cursor.generate(
                    "argument may not have 'value' and 'dimension' attribute."
                )
            if rank is not missing:
                self.cursor.generate(
                    "argument may not have 'rank' and 'dimension' attribute."
                )
            if not is_ptr:
                self.cursor.generate(
                    "dimension attribute can only be "
                    "used on pointer and references"
                )
            self.parse_dim_attrs(dimension, meta)
        elif ntypemap:
            if ntypemap.implied_array:
                # default to 1-d assumed shape
                meta["rank"] = 1

        owner = attrs.get("owner", missing)
        if owner is not missing:
            # XXX - Need to extract smart_poiner from Typemaps
            if owner not in ["caller", "library", "shared", "weak"]:
                # XXX - shared is only valued with language=c++
                self.cursor.generate(
                    "Illegal value '{}' for owner attribute. "
                    "Must be 'caller' or 'library'.".format(owner)
                )
            meta["owner"] = owner

        free_pattern = attrs.get("free_pattern", missing)
        if free_pattern is not missing:
            if free_pattern not in self.newlibrary.patterns:
                raise RuntimeError(
                    "Illegal value '{}' for free_pattern attribute. "
                    "Must be defined in patterns section.".format(free_pattern)
                )
            meta["free_pattern"] = free_pattern

        lenattr = attrs.get("len", missing)
        if lenattr is not missing:
            meta["len"] = lenattr
            
    def parse_dim_attrs(self, dim, meta):
        """Parse dimension attributes and save the AST.
        This tree will be traversed by the wrapping classes
        to convert to language specific code.

        Parameters
        ----------
        dim : dimension string
        meta: Scope
        """
        if not dim:
            return
        try:
            meta["dimension"] = dim
            meta["dim_ast"] = declast.check_dimension(dim)
        except error.ShroudParseError:
            self.cursor.generate("Unable to parse dimension: {}"
                                     .format(dim))

######################################################################
#

class FillMetaC(FillMeta):
    def meta_typedef(self, cls, node):
        # node - ast.TypedefNode
        wlang = self.wlang
        func_cursor = self.cursor.current
        #####
        r_bind = statements.fetch_typedef_bind(node, wlang)
        self.set_typedef_share(node, r_bind.meta)

    def meta_variable(self, cls, node):
        wlang = self.wlang
        node_cursor = self.cursor.current
        
    def meta_function(self, cls, node):
        if not node.wrap.c:
            return
        wlang = self.wlang
        func_cursor = self.cursor.current
        #####
        declarator = node.ast.declarator

        r_bind = statements.fetch_func_bind(node, wlang)
        r_meta = r_bind.meta

        self.set_func_share(node, r_meta)
        self.set_func_deref_c(node, r_meta)
        self.set_func_api_c(node, r_meta)
        self.set_func_post_c(cls, node, r_meta)

        # --- Loop over function parameters
        for arg in declarator.params:
            func_cursor.arg = arg
            a_bind = statements.fetch_arg_bind(node, arg, wlang)
            meta = a_bind.meta

            self.set_arg_share(node, arg, meta)
            self.set_arg_deref_c(arg, meta)
            self.set_arg_api_c(arg, meta)
        # --- End loop over function parameters
        func_cursor.arg = None

        # Lookup statements if there are no meta attribute errors
        if node.wrap.c:
            stmt0 = statements.lookup_c_function_stmt(node)
            result_stmt = statements.lookup_local_stmts([wlang], stmt0, node)
            r_bind.stmt = result_stmt
            if stmt0 is not result_stmt:
                r_bind.fstmts = wlang
            for arg in declarator.params:
                arg_stmt = statements.lookup_c_arg_stmt(node, arg)
                a_bind = statements.get_arg_bind(node, arg, wlang)
                a_bind.stmt = arg_stmt
            
######################################################################
#

class FillMetaFortran(FillMeta):
    def meta_typedef(self, cls, node):
        # node - ast.TypedefNode
        wlang = self.wlang
        func_cursor = self.cursor.current
        #####
        meta = statements.fetch_typedef_bind(node, wlang).meta
        self.set_typedef_share(node, meta)

        arg = node.ast
        if arg.declarator.is_function_pointer():
            fptr = meta["fptr"]
            statements.fetch_func_bind(fptr, wlang)
            self.meta_function(None, fptr)

    def meta_variable(self, cls, node):
        wlang = self.wlang
        node_cursor = self.cursor.current
        
    def meta_function(self, cls, node, fptr_arg=False):
        if not node.wrap.fortran:
            return
        wlang = self.wlang
        func_cursor = self.cursor.current
        #####
        ast = node.ast
        declarator = ast.declarator

        r_bind = statements.fetch_func_bind(node, wlang)
        r_meta = r_bind.meta

        self.set_func_share(node, r_meta)
        self.set_func_deref_fortran(node, r_meta)
        self.set_func_api_fortran(node, r_meta)
        self.set_func_post_fortran(cls, node, r_meta)
        
        self.meta_function_params(node, fptr_arg)

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

    def meta_function_params(self, node, fptr_arg=False):
        wlang = self.wlang
        func_cursor = self.cursor.current
        for arg in node.ast.declarator.params:
            func_cursor.arg = arg

            a_bind = statements.fetch_arg_bind(node, arg, wlang)
            meta = a_bind.meta

            self.set_arg_share(node, arg, meta)
            self.set_arg_fortran(node, arg, meta)
            self.set_arg_deref_fortran(node, arg, meta)
            self.set_arg_api_fortran(node, arg, meta, fptr_arg)
            self.set_arg_hidden(arg, meta)

            if arg.declarator.is_function_pointer():
                fptr = meta["fptr"]
                self.meta_function(None, fptr, fptr_arg=True)
        func_cursor.arg = None
        
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
            
        blanknull = attrs.get("blanknull", missing)
        if blanknull is not missing:
            if not char_ptr_in:
                self.cursor.generate(
                    "blanknull attribute can only be "
                    "used on intent(in) 'char *'"
                )
        elif char_ptr_in:
            blanknull = options.F_blanknull
        if blanknull is True:
            meta["blanknull"] = blanknull

        if "api" in attrs:  # User set
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
    def meta_typedef(self, cls, node):
        pass

    def meta_variable(self, cls, node):
        wlang = self.wlang
        node_cursor = self.cursor.current
        
    def meta_function(self, cls, node):
        if not node.wrap.python:
            return
        wlang = self.wlang
        func_cursor = self.cursor.current
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

            a_bind = statements.fetch_arg_bind(node, arg, wlang)
            meta = a_bind.meta

            self.set_arg_share(node, arg, meta)
            self.set_arg_deref(arg, meta)
            self.set_arg_hidden(arg, meta)
            arg_stmt = None
            a_bind.stmt = arg_stmt

        # --- End loop over function parameters
        func_cursor.arg = None

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

        if "deref" in attrs:
            deref = self.filter_deref(attrs["deref"])
            if deref:
                meta["deref"] = deref

    def set_arg_deref(self, arg, meta):
        declarator = arg.declarator
        attrs = declarator.attrs

        if "deref" in attrs:
            deref = self.filter_deref(attrs["deref"])
            if deref:
                meta["deref"] = deref
        
######################################################################
#

class FillMetaLua(FillMeta):
    def meta_typedef(self, cls, node):
        pass

    def meta_variable(self, cls, node):
        wlang = self.wlang
        node_cursor = self.cursor.current
        
    def meta_function(self, cls, node):
        if not node.wrap.python:
            return
        wlang = self.wlang
        func_cursor = self.cursor.current
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

            a_bind = statements.fetch_arg_bind(node, arg, wlang)
            meta = a_bind.meta

            self.set_arg_share(node, arg, meta)
            arg_stmt = None
            a_bind.stmt = arg_stmt

        # --- End loop over function parameters
        func_cursor.arg = None

######################################################################
#

def dereference_function_pointer(declaration):
    """Return a new Declaration with dereferenced function pointer.

    Only pointer to function - no pointer to pointer to function.
    """
    declarator = declaration.declarator
    if not declarator.is_function_pointer():
        raise RuntimeError("Expected a function pointer")
    if declarator.func.is_pointer() != 1:
        raise RuntimeError("Expected single pointer to function")
    newdecl = copy.copy(declaration)
    newdecl.declarator = copy.copy(declarator)
    newdecl.declarators = [ newdecl.declarator ]
    newdecl.declarator.func = None
    return newdecl
    
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
    """Process attributes for a language.
    """
    process_map[wlang](newlibrary, wlang).meta_library()
        
