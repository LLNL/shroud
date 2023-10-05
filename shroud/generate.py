# Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and
# other Shroud Project Developers.
# See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

"""
Check attributes in delcaration.
Set meta attributes.
Generate additional functions required to create wrappers.
"""
from __future__ import print_function
from __future__ import absolute_import

import collections
import copy

from . import ast
from . import declast
from . import error
from . import todict
from . import statements
from . import typemap
from . import util
from . import visitor
from . import whelpers

wformat = util.wformat


class VerifyAttrs(object):
    """
    Check attributes and set some defaults in metaattrs.
    Generate types for classes.

    check order:
      intent - check_intent_attr
      deref - check_deref_attr_func / check_deref_attr_var
    """

    def __init__(self, newlibrary, config):
        """
        Args:
            newlibrary - ast.LibraryNode
            config -
        """
        self.newlibrary = newlibrary
        self.config = config
        self.cursor = error.get_cursor()

    def verify_attrs(self):
        """Verify library attributes.
        Recurse through all declarations.
        Entry pointer for class VerifyAttrs.
        """
        self.cursor.push_phase("verify attributes")
        self.verify_namespace_attrs(self.newlibrary.wrap_namespace)
        self.cursor.pop_phase("verify attributes")

    def verify_namespace_attrs(self, node):
        """Verify attributes for a library or namespace.

        Args:
            node - ast.LibraryNode, ast.NameSpaceNode
        """
        cursor = self.cursor
        for cls in node.classes:
            cursor.push_node(cls)
            for var in cls.variables:
                cursor.push_node(var)
                self.check_var_attrs(cls, var)
                cursor.pop_node(var)
            for func in cls.functions:
                cursor.push_node(func)
                self.check_fcn_attrs(func)
                cursor.pop_node(func)
            cursor.pop_node(cls)

        for func in node.functions:
            cursor.push_node(func)
            self.check_fcn_attrs(func)
            cursor.pop_node(func)

        for ns in node.namespaces:
            cursor.push_node(ns)
            self.verify_namespace_attrs(ns)
            cursor.pop_node(ns)

    def check_var_attrs(self, cls, node):
        """Check attributes for variables.
        This includes struct and class members.

        Args:
            cls -
            node -
        """
        ast = node.ast
        declarator = ast.declarator
        attrs = declarator.attrs
        for attr in attrs:
            if attr[0] == "_":  # internal attribute
                continue
            # XXX - deref on class/struct members
            if attr not in ["name", "readonly", "dimension", "deref"]:
                self.cursor.generate(
                    "Illegal attribute '{}' for variable '{}'".format(
                        attr, node.name
                    ) + "\nonly 'name', 'readonly', 'dimension' and 'deref' are allowed on variables"
                )

        dim = attrs["dimension"]
        if dim:
            is_ptr = declarator.is_indirect()
            if not is_ptr:
                self.cursor.generate(
                    "dimension attribute can only be "
                    "used on pointer and references"
                )
            meta = declarator.metaattrs
            self.parse_dim_attrs(dim, meta)

    def check_fcn_attrs(self, node):
        """Check attributes on FunctionNode.

        Args:
            node - ast.FunctionNode
        """
        cursor = self.cursor
        options = node.options

        ast = node.ast
        declarator = ast.declarator
        attrs = declarator.attrs
        meta = declarator.metaattrs
        node._has_found_default = False

        for attr in attrs:
            if attr[0] == "_":  # internal attribute
                continue
            if attr not in [
                "api",          # arguments to pass to C wrapper.
                "allocatable",  # return a Fortran ALLOCATABLE
                "deref",  # How to dereference pointer
                "dimension",
                "free_pattern",
                "len",
                "name",
                "owner",
                "pure",
                "rank",
            ]:
                cursor.generate(
                    "Illegal attribute '{}' for function '{}'".format(
                        attr, node.name
                    )
                )

        if ast.typemap is None:
            print("XXXXXX typemap is None")
        if ast.typemap.sgroup == "shadow":
            if options.C_shadow_result:
                meta["api"] = "capptr"
            else:
                meta["api"] = "capsule"
        if declarator.is_ctor():
            meta["intent"] = "ctor"
        elif declarator.is_dtor():
            meta["intent"] = "dtor"
        else:
            meta["intent"] = declarator.get_subprogram()
        self.check_deref_attr_func(node)
        self.check_common_attrs(node.ast)

        for arg in declarator.params:
            if arg.declarator.name is None:
                cursor.generate("Argument must have name in {}".format(
                    node.decl))
            self.check_arg_attrs(node, arg)

        if node.fortran_generic:
            for generic in node.fortran_generic:
                for garg in generic.decls:
                    generic._has_found_default = False
                    self.check_arg_attrs(generic, garg, node.options)
                check_implied_attrs(node, generic.decls)
        else:
            check_implied_attrs(node, declarator.params)

    def check_intent_attr(self, node, arg):
        """Set default intent meta-attribute.

        Intent is only valid on arguments.
        intent: lower case, no parens, must be in, out, or inout
        """
        declarator = arg.declarator
        attrs = declarator.attrs
        meta = declarator.metaattrs
        is_ptr = declarator.is_indirect()
        intent = attrs["intent"]
        if intent is None:
            if node is None:
                # do not default intent for function pointers
                pass
            elif declarator.is_function_pointer():
                intent = "in"
            elif not is_ptr:
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
            if intent not in ["in", "out", "inout"]:
                self.cursor.generate("Bad value for intent: " + attrs["intent"])
                intent = "inout"
            elif not is_ptr and intent != "in":
                # Nonpointers can only be intent(in).
                self.cursor.generate("Only pointer arguments may have intent of 'out' or 'inout'")
        meta["intent"] = intent
        return intent    
        
    def check_deref_attr_func(self, node):
        """Check deref attr and set default for function.

        Function which return pointers or objects (std::string)
        set the deref meta attribute.

        Parameters
        ----------
        node : FunctionNode
        ast : declast.Declaration
        """
        options = node.options
        ast = node.ast
        declarator = ast.declarator
        attrs = declarator.attrs
        deref = attrs["deref"]
        mderef = None
        ntypemap = ast.typemap
        nindirect = declarator.is_indirect()
        attrs = ast.declarator.attrs
        meta = ast.declarator.metaattrs
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
                mderef = options.return_scalar_pointer
        elif deref:
            self.cursor.generate("Cannot have attribute 'deref' on non-pointer function")
        meta["deref"] = mderef
        
    def check_deref_attr_var(self, node, ast):
        """Check deref attr and set default for variable.

        Pointer variables set the default deref meta attribute.

        Parameters
        ----------
        node - ast.FunctionNode or ast.FortranGeneric
        ast : declast.Declaration
        """
        declarator = ast.declarator
        attrs = declarator.attrs
        meta = declarator.metaattrs
        ntypemap = ast.typemap
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
            
    def check_common_attrs(self, ast):
        """Check attributes which are common to function and argument AST
        This includes: dimension, free_pattern, owner, rank

        Parameters
        ----------
        ast : declast.Declaration
        """
        declarator = ast.declarator
        attrs = declarator.attrs
        meta = declarator.metaattrs
        ntypemap = ast.typemap
        is_ptr = declarator.is_indirect()

        # api
        api = attrs["api"]
        if api is None:
            pass
        elif api not in ["capi", "buf", "cdesc", "cfi"]:
                self.cursor.generate(
                    "'api' attribute must 'capi', 'buf', 'cdesc' or 'cfi'"
                )
        else:
            meta["api"] = api

        # dimension
        dimension = attrs["dimension"]
        rank = attrs["rank"]
        if rank:
            if rank is True:
                self.cursor.generate(
                    "'rank' attribute must have an integer value"
                )
            else:
                try:
                    attrs["rank"] = int(attrs["rank"])
                except ValueError:
                    self.cursor.generate(
                        "rank attribute must have an integer value, not '{}'"
                        .format(attrs["rank"])
                    )
                else:
                    if attrs["rank"] > 7:
                        self.cursor.generate(
                            "'rank' attribute must be 0-7, not '{}'"
                            .format(attrs["rank"])
                        )
            if not is_ptr:
                self.cursor.generate(
                    "rank attribute can only be "
                    "used on pointer and references"
                )
        if dimension:
            if dimension is True:
                self.cursor.generate(
                    "dimension attribute must have a value."
                )
                dimension = None
            if attrs["value"]:
                self.cursor.generate(
                    "argument may not have 'value' and 'dimension' attribute."
                )
            if rank:
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
            if ntypemap.base == "vector":
                # default to 1-d assumed shape
                attrs["rank"] = 1
            elif ntypemap.name == 'char' and is_ptr == 2:
                # 'char **' -> CHARACTER(*) s(:)
                attrs["rank"] = 1

        owner = attrs["owner"]
        if owner is not None:
            if owner not in ["caller", "library"]:
                self.cursor.generate(
                    "Illegal value '{}' for owner attribute. "
                    "Must be 'caller' or 'library'.".format(owner)
                )

        free_pattern = attrs["free_pattern"]
        if free_pattern is not None:
            if free_pattern not in self.newlibrary.patterns:
                raise RuntimeError(
                    "Illegal value '{}' for free_pattern attribute. "
                    "Must be defined in patterns section.".format(free_pattern)
                )

    def check_arg_attrs(self, node, arg, options=None):
        """Regularize attributes.
        value: if pointer, default to None (pass-by-reference)
               else True (pass-by-value).

        When used for fortran_generic, pass in node's options
        since FunctionNode for each generic is not yet created
        via GenFunctions.generic_function.

        options are also passed in for function pointer arguments
        since node will be None.

        Args:
            node - ast.FunctionNode or ast.FortranGeneric
            arg  - declast.Declaration
            options -
        """
        cursor = self.cursor
        if options is None:
            options = node.options
        declarator = arg.declarator
        argname = declarator.user_name
        attrs = declarator.attrs
        meta = declarator.metaattrs

        for attr in attrs:
            if attr[0] == "_":  # Shroud internal attribute.
                continue
            if attr not in [
                "api",
                "allocatable",
                "assumedtype",
                "blanknull",   # Treat blank string as NULL pointer.
                "charlen",   # Assumed length of intent(out) char *.
                "external",
                "deref",
                "dimension",
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

        intent = self.check_intent_attr(node, arg)
        self.check_deref_attr_var(node, arg)
        self.check_common_attrs(arg)

        is_ptr = declarator.is_indirect()

        # assumedtype
        assumedtype = attrs["assumedtype"]
        if assumedtype is not None:
            if attrs["value"]:
                cursor.generate(
                    "argument '{}' must not have value=True "
                    "because it has the assumedtype attribute.".format(argname)
                )

        # value
        elif attrs["value"] is None:
            if is_ptr:
                if arg_typemap.name == "void":
                    # This causes Fortran to dereference the C_PTR
                    # Otherwise a void * argument becomes void **
                    if len(arg.declarator.pointer) == 1:
                        attrs["value"] = True  # void *
#                    else:
#                        attrs["value"] = None # void **  XXX intent(out)?
            else:
                attrs["value"] = True

        # charlen
        # Only meaningful with 'char *arg+intent(out)'
        # XXX - Python needs a value if 'char *+intent(out)'
        charlen = attrs["charlen"]
        if charlen:
            if arg_typemap.base != "string":
                cursor.generate(
                    "charlen attribute can only be "
                    "used on 'char *'"
                )
            elif is_ptr != 1:
                cursor.generate(
                    "charlen attribute can only be "
                    "used on 'char *'"
                )
            elif charlen is True:
                cursor.generate("charlen attribute must have a value")

        char_ptr_in = (
            is_ptr == 1 and
            intent == "in" and
            arg_typemap.name == "char")
            
        blanknull = attrs["blanknull"]
        if blanknull is not None:
            if not char_ptr_in:
                cursor.generate(
                    "blanknull attribute can only be "
                    "used on intent(in) 'char *'"
                )
            else:
                arg.blanknull = blanknull
        elif char_ptr_in:
            arg.blanknull = options.F_blanknull

        if meta["api"]:  # User set
            pass
        elif (
            options.F_CFI is False and
            char_ptr_in and
            arg.blanknull is False
        ):
            # const char *arg
            # char *arg+intent(in)
            # Add terminating NULL in Fortran wrapper.
            # Avoid a C wrapper just to do the NULL terminate.
            arg.ftrim_char_in = options.F_trim_char_in

        if node:
            if declarator.init is not None:
                node._has_default_arg = True
            elif node._has_found_default is True:
                raise RuntimeError("Expected default value for %s" % argname)

        # Check template attribute
        # XXX - This should be part of typemap
        temp = arg.template_arguments
        if arg_typemap and arg_typemap.base == "vector":
            if not temp:
                cursor.generate(
                    "std::vector must have template argument: {}".format(
                        arg.gen_decl()
                    )
                )
            else:
                arg_typemap = arg.template_arguments[0].typemap
                if arg_typemap is None:
                    # XXX - Not sure this can happen with current parser
                    raise RuntimeError(
                        "check_arg_attr: No such type %s for template: %s"
                        % (temp, arg.gen_decl())
                    )
        elif temp:
            raise RuntimeError(
                "Type '%s' may not supply template argument: %s"
                % (arg_typemap.name, arg.gen_decl())
            )

        # Flag node if any argument is assumed-rank.
        if meta["assumed-rank"]:
            node._gen_fortran_generic = True

        if declarator.is_function_pointer():
            for arg1 in declarator.params:
                self.check_arg_attrs(None, arg1, options)

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
            check_dimension(dim, meta)
        except error.ShroudParseError:
            self.cursor.generate("Unable to parse dimension: {}"
                                     .format(dim))


def check_dimension(dim, meta, trace=False):
    """Assign AST of dim and assumed_rank flag to meta.

    Look for assumed-rank, "..", first.
    Else a comma delimited list of expressions.

    Parameters
    ----------
    dim : str
    meta : dict
    trace : boolean
    """
    if dim == "..":
        meta["dimension"] = declast.AssumedRank()
        meta["assumed-rank"] = True
    else:
        meta["dimension"] = declast.ExprParser(dim, trace=trace).dimension_shape()


class GenFunctions(object):
    """
    Generate Typemap from class.
    Generate functions based on overload/template/generic/attributes
    Computes fmt.function_suffix.

    gen_library
      instantiate_all_classes
        instantiate_classes
          add_struct_ctor
          process_class
            add_var_getter_setter
            define_function_suffix
              has_default_args
              template_function
              define_fortran_generic_functions
              define_bufferify_functions
                arg_to_CFI
                arg_to_buffer
      update_templated_typemaps
      gen_namespace
        define_function_suffix
    """

    def __init__(self, newlibrary, config):
        """
        Args:
            newlibrary - ast.LibraryNode
            config -
        """
        self.newlibrary = newlibrary
        self.config = config
        self.instantiate_scope = None
        self.language = newlibrary.language
        self.cursor = error.get_cursor()

    def gen_library(self):
        """Entry routine to generate functions for a library.
        """
        self.cursor.push_phase("generate functions")
        newlibrary = self.newlibrary
        whelpers.add_all_helpers(newlibrary.symtab)

        self.function_index = newlibrary.function_index
        self.class_map = newlibrary.class_map

        self.instantiate_all_classes(newlibrary.wrap_namespace)
        self.update_templated_typemaps(newlibrary.wrap_namespace)
        self.gen_namespace(newlibrary.wrap_namespace)
        self.cursor.pop_phase("generate functions")

    def gen_namespace(self, node):
        """Process functions which are not in a class.

        Args:
            node - ast.LibraryNode, ast.NamespaceNode
        """
        node.functions = self.define_function_suffix(node.functions)
        for ns in node.namespaces:
            self.gen_namespace(ns)

    # No longer need this, but keep code for now in case some other dependency checking is needed
    #        for cls in newlibrary.classes:
    #            self.check_class_dependencies(cls)

    def push_instantiate_scope(self, node, targs):
        """Add template arguments to scope.

        Args:
            node  - ClassNode or FunctionNode
            targs - ast.TemplateArgument
        """
        newscope = util.Scope(self.instantiate_scope)
        for idx, argast in enumerate(targs.asts):
            setattr(newscope, node.template_parameters[idx], argast)
        self.instantiate_scope = newscope

    def pop_instantiate_scope(self):
        """Remove template arguments from scope"""
        self.instantiate_scope = self.instantiate_scope.get_parent()

    def append_function_index(self, node):
        """append to function_index, set index into node.

        self.function index is the LibraryNode.function_index.

        Args:
            node -
        """
        ilist = self.function_index
        node._function_index = len(ilist)
        #        node.fmtdict.function_index = str(len(ilist)) # debugging
        ilist.append(node)

    def XXX_default_ctor_and_dtor(self, cls):
        """Wrap default constructor and destructor.

        Needed when the ctor or dtor is not explicily in the input.

        XXX - do not wrap by default because they may be private.
        """
        found_ctor = False
        found_dtor = False
        for node in cls.functions:
            declarator = node.ast.declarator
            found_ctor = found_ctor or declarator.is_ctor()
            found_dtor = found_dtor or declarator.is_dtor()

        if found_ctor and found_dtor:
            return cls.functions

        added = cls.functions[:]

        if not found_ctor:
            added.append(ast.FunctionNode("{}()".format(cls.name), parent=cls))
        if not found_dtor:
            added.append(ast.FunctionNode("~{}()".format(cls.name), parent=cls))

        return added

    def add_var_getter_setter(self, parent, cls, var):
        """Create getter/setter functions for class variables.
        This allows wrappers to access class members.

        Do not wrap for Python since descriptors are created for
        class member variables.

        The getter is a function of the same type as var
        with no arguments.
        The setter is a void function with a single argument
        the same type as var.

        Generated functions are added to the parent node.
        For a class, they're added to the class.
        For a struct, they're added to the struct container (Library, Namespace)

        Must explicitly set metaattrs for arguments since that was
        done earlier when validating attributes.

        The 'struct' meta attribute is set on the getter so member
        names in the dimension attribute can be looked up. And set on
        the 'this' argument to help set CXX_this properly in the C
        wrapper.

        Parameters
        ----------
        parent : ast.LibraryNode, ast.ClassNode
        cls : ast.ClassNode
        var : ast.VariableNode

        """
        options = var.options
        if options.wrap_fortran is False and options.wrap_c is False:
            return

        ast = var.ast
        declarator = ast.declarator
        sgroup = ast.typemap.sgroup

        fmt = util.Scope(var.fmtdict)
        fmt_func = dict(
            # Use variable's field_name for the generated functions.
            field_name=var.fmtdict.field_name, # Name of member variable
            wrapped_name=declarator.user_name, # Using name attribute
            struct_name=cls.fmtdict.cxx_type,
        )

        is_struct = cls.wrap_as == "struct"
        if is_struct:
            if not options.F_struct_getter_setter:
                return
            if declarator.is_pointer() != 1:
                # Skip scalar and char**.
                return
            if sgroup in ["char", "string"]:
                # No strings for now.
                return
            # Explicity add the 'this' argument. Added automatically for classes.
            typename = cls.typemap.name
            this_get = "{} *{}".format(typename, cls.fmtdict.CXX_this)
            this_set = this_get + ","
            funcname_get = wformat(options.SH_struct_getter_template, fmt_func)
            funcname_set = wformat(options.SH_struct_setter_template, fmt_func)
        else:
            this_get = ""
            this_set = ""
            funcname_get = wformat(options.SH_class_getter_template, fmt_func)
            funcname_set = wformat(options.SH_class_setter_template, fmt_func)

        if self.language == "c":
            lang = "c_type"
        else:
            lang = "cxx_type"

        deref = None
        if sgroup in ["char", "string"]:
            value = None
            deref = "allocatable"
        elif sgroup == "vector":
            value = None
            deref = "pointer"
        elif declarator.is_pointer():
            value = None
            deref = "pointer"
        else:
            value = True
            deref = None

        ##########
        # getter
        argdecl = ast.gen_arg_as_language(lang=lang, name=funcname_get, continuation=True)
        decl = "{}({})".format(argdecl, this_get)

        fattrs = {}

        fcn = parent.add_function(decl, format=fmt_func, fattrs=fattrs)
        meta = fcn.ast.declarator.metaattrs
        meta.update(declarator.metaattrs)
        meta["intent"] = "getter"
        meta["deref"] = deref
        if is_struct:
            meta["struct"] = cls.typemap.flat_name
            meta = fcn.ast.declarator.params[0].declarator.metaattrs
            meta["struct"] = cls.typemap.flat_name
            meta["intent"] = "in"
        fcn.wrap.lua = False
        fcn.wrap.python = False
        fcn._generated = "getter/setter"

        ##########
        # setter
        if declarator.attrs["readonly"]:
            return
        argdecl = ast.gen_arg_as_language(lang=lang, name="val", continuation=True)
        decl = "void {}({}{})".format(funcname_set, this_set, argdecl)

        attrs = dict(
            val=dict(
                intent="in",
                value=value,
            )
        )
        dim = declarator.metaattrs["dimension"]
        if dim:
            attrs["val"]["rank"] = len(dim)

        fcn = parent.add_function(decl, attrs=attrs, format=fmt_func)
        # XXX - The function is not processed like other, so set intent directly.
        fcn.ast.declarator.metaattrs["intent"] = "setter"
        iarg = 0
        if is_struct:
            meta = fcn.ast.declarator.params[0].declarator.metaattrs
            meta["intent"] = "inout"
            meta["struct"] = cls.typemap.flat_name
            iarg = 1
        meta = fcn.ast.declarator.params[iarg].declarator.metaattrs
        meta.update(declarator.metaattrs)
        meta["intent"] = "setter"
        fcn.wrap.lua = False
        fcn.wrap.python = False
        fcn._generated = "getter/setter"

    def instantiate_all_classes(self, node):
        """Instantate all class template_arguments recursively.

        Args:
            node - ast.LibraryNode, ast.NamespaceNode, ast.ClassNode
        """
        self.instantiate_classes(node)

        for cls in node.classes:
            self.instantiate_classes(cls)

        for ns in node.namespaces:
            self.instantiate_all_classes(ns)

    def instantiate_classes(self, node):
        """Instantate any template_arguments.

        Create a new list of classes replacing
        any class with template_arguments with instantiated classes.
        All new classes will be added to node.classes.

        Parameters
        ----------
        node : ast.LibraryNode, ast.NamespaceNode, ast.ClassNode
        """
        clslist = []
        for cls in node.classes:
            self.cursor.push_node(cls)
            if cls.wrap_as == "struct":
                clslist.append(cls)
                options = cls.options
                if cls.wrap.python and options.PY_struct_arg == "class":
                    self.add_struct_ctor(cls)
                self.process_class(node, cls)
            elif cls.template_arguments:
                orig_typemap = cls.typemap
                if orig_typemap.cxx_instantiation is None:
                    orig_typemap.cxx_instantiation = {}
                # Replace class with new class for each template instantiation.
                # targs -> ast.TemplateArgument
                for i, targs in enumerate(cls.template_arguments):
                    newcls = cls.clone()
                    clslist.append(newcls)

                    # If single template argument, use its name; else sequence.
                    # XXX - maybe change to names
                    #   i.e.  _int_double  However <std::string,int> is a problem.
                    if targs.fmtdict and 'template_suffix' in targs.fmtdict:
                        class_suffix = targs.fmtdict['template_suffix']
                    elif len(targs.asts) == 1:
                        ntypemap = targs.asts[0].typemap
                        if ntypemap.template_suffix:
                            class_suffix = ntypemap.template_suffix
                        else:
                            class_suffix = "_" + ntypemap.flat_name
                    else:
                        class_suffix = "_" + str(i)

                    # Update name of class.
                    #  name_api           - vector_0 or vector_int     (Fortran and C names)
                    #  name_instantiation - vector<int>
                    if targs.fmtdict and "cxx_class" in targs.fmtdict:
                        newcls.name_api = targs.fmtdict["cxx_class"]
                    else:
                        newcls.name_api = cls.name + class_suffix
                    newcls.name_instantiation = cls.name + targs.instantiation
                    newcls.scope_file[-1] += class_suffix

                    if targs.fmtdict:
                        newcls.user_fmt.update(targs.fmtdict)
                    if targs.options:
                        newcls.options.update(targs.options)
                    
                    # Remove defaulted attributes then reset with current values.
                    newcls.delete_format_templates()
                    newcls.default_format()

                    newcls.typemap = typemap.create_class_typemap(newcls)
                    if targs.instantiation in orig_typemap.cxx_instantiation:
                        print("instantiate_classes: {} already in "
                              "typemap.cxx_instantiation".format(targs.instantiation))
                    orig_typemap.cxx_instantiation[targs.instantiation] = newcls.typemap

                    self.template_typedef(newcls, targs)

                    self.push_instantiate_scope(newcls, targs)
                    self.process_class(newcls, newcls)
                    self.pop_instantiate_scope()
            else:
                clslist.append(cls)
                self.process_class(cls, cls)
            self.cursor.pop_node(cls)

        node.classes = clslist

    def template_typedef(self, node, targs):
        """Create a new typemap for instantiated templated typedefs.

        Replace typemap in function arguments with
        class instantiation of the typemap.

        node -> ClassNode
        """
        typedefmap = []
        for typ in node.typedefs:
            oldtyp = typ.typemap
            typ.clone_post_class(targs)
            typedefmap.append( (oldtyp, typ.typemap) )
        node.typedef_map = typedefmap

        for function in node.functions:
            for arg in function.ast.declarator.params:
                ntypemap = arg.typemap
                for typedef in typedefmap:
                    if ntypemap is typedef[0]:
                        arg.typemap = typedef[1]
                        break
            
    def update_templated_typemaps(self, node):
        """Update templated types to use correct typemap.

        Each templated class must be instantated in the YAML type.
        """
        visitor = TemplateTypemap(self.config)
        return visitor.visit(node)
        
    def add_struct_ctor(self, cls):
        """Add a Python constructor function for a struct when
        it will be treated like a class.

        Args:
            cls - ast.ClassNode
        """
        ast = declast.create_struct_ctor(cls)
        name = ast.declarator.attrs["name"]  #cls.name + "_ctor"
        #  Add variables as function parameters by coping AST.
        for var in cls.variables:
            a = copy.deepcopy(var.ast)
            a.declarator.metaattrs["intent"] = "in"
            a.declarator.metaattrs["struct_member"] = var
            ast.declarator.params.append(a)
        # Python only
        opt = dict(
            wrap_fortran=False,
            wrap_c=False,
            wrap_lua=False,
        )
        node = cls.add_function(name, ast, options=opt)
        node.declgen = node.ast.gen_decl()
        node._generated = "struct_as_class_ctor"

    def process_class(self, parent, cls):
        """Process variables and functions for a class/struct.
        Create getter/setter functions for member variables.

        Parameters
        ----------
        parent : ast.LibraryNode, ast.ClassNode
        cls : ast.ClassNode
        """
        if cls.typemap.flat_name in self.class_map:
            self.cursor.generate("process_class: class {} already exists in class_map"
                               .format(cls.typemap.flat_name))
            return
        self.class_map[cls.typemap.flat_name] = cls
        for var in cls.variables:
            self.add_var_getter_setter(parent, cls, var)
        cls.functions = self.define_function_suffix(cls.functions)

    def define_bufferify_functions(self, functions):
        """Create additional C bufferify functions."""
        ordered = []
        for node in functions:
            ordered.append(node)
            
            if node.options.F_CFI:
#                node._gen_fortran_generic = False
                done = self.arg_to_CFI(node, ordered)
            else:
                done = False

            if node.options.F_create_bufferify_function and not done:
                self.arg_to_buffer(node, ordered)
        return ordered

    def define_fortran_generic_functions(self, functions):
        """Create multiple generic Fortran wrappers to call a
        single C function.
        """
        ordered = []
        for node in functions:
            ordered.append(node)
            if not node.wrap.fortran:
                continue
            if node._gen_fortran_generic and not node.options.F_CFI:
                self.process_assumed_rank(node)
            if node.fortran_generic:
                node._overloaded = True
                self.generic_function(node, ordered)
        return ordered
        
    def define_function_suffix(self, functions):
        """Return a new list with generated function inserted.

        Parameters
        ----------
        functions : list of ast.FunctionNode
        """

        # Look for overloaded functions
        cxx_overload = {}
        for function in functions:
            self.append_function_index(function)
            cxx_overload.setdefault(function.name, []).append(
                function._function_index
            )

        # keep track of which function are overloaded in C++.
        for value in cxx_overload.values():
            if len(value) > 1:
                for index in value:
                    self.function_index[index]._cxx_overload = value

        # Create additional functions needed for wrapping
        ordered_functions = []
        for method in functions:
            if method._has_default_arg:
                self.has_default_args(method, ordered_functions)
            ordered_functions.append(method)
            if method.template_arguments:
                method._overloaded = True
                self.template_function(method, ordered_functions)
            elif method.have_template_args:
                # have_template_args is True if result/argument is templated.
                #                method._overloaded = True
                self.template_function2(method, ordered_functions)

        # Look for overloaded functions
        overloaded_functions = {}
        for function in ordered_functions:
            # if not function.wrap.c:
            #     continue
            if function.cxx_template:
                continue
            if function.template_arguments:
                continue
            if function.have_template_args:
                # Stuff like push_back which is in a templated class, is not an overload.
                # C_name_scope is used to distigunish the functions, not function_suffix.
                continue
            if function.ast.declarator.is_ctor():
                if not function.wrap.fortran:
                    continue
                # Always create generic interface for class derived type.
                fmt = function.fmtdict
                name = fmt.F_derived_name
                overloaded_functions.setdefault(name, []).append(
                    function)
                function.options.F_create_generic = True
                fmt.F_name_generic = name
                function._overloaded = True
            else:
                overloaded_functions.setdefault(function.name, []).append(
                    function)

        # look for function overload and compute function_suffix
        for overloads in overloaded_functions.values():
            if len(overloads) > 1:
                for i, function in enumerate(overloads):
                    function._overloaded = True
                    if not function.fmtdict.inlocal("function_suffix"):
                        function.fmtdict.function_suffix = "_{}".format(i)

        # return_this
        ordered2 = []
        for method in ordered_functions:
            ordered2.append(method)
            if method.return_this:
                self.process_return_this(method, ordered2)
                
        ordered3 = self.define_fortran_generic_functions(ordered2)
        ordered4 = self.define_bufferify_functions(ordered3)

        self.gen_functions_decl(ordered4)

        return ordered4

    def template_function(self, node, ordered_functions):
        """ Create overloaded functions for each templated argument.

        - decl: template<typename ArgType> void Function7(ArgType arg)
          cxx_template:
          - instantiation: <int>
          - instantiation: <double>
            format:
              template_suffix: dbl

        node.template_arguments = [ TemplateArgument('<int>'), TemplateArgument('<double>')]
                 TemplateArgument.asts[i].typemap

        Clone entire function then look for template arguments.

        Args:
            node -
            ordered_functions -
        """
        oldoptions = node.options
        headers_typedef = collections.OrderedDict()

        # targs - ast.TemplateArgument
        for iargs, targs in enumerate(node.template_arguments):
            new = node.clone()
            ordered_functions.append(new)
            self.append_function_index(new)

            new._generated = "cxx_template"

            fmt = new.fmtdict
            if targs.fmtdict:
                fmt.update(targs.fmtdict)

            # Use explicit template_suffix if provide.
            # If single template argument, use type's explicit_suffix
            # or the unqualified flat_name.
            # Multiple template arguments, use sequence number.
            if fmt.template_suffix:
                pass
            elif len(targs.asts) == 1:
                ntypemap = targs.asts[0].typemap
                if ntypemap.template_suffix:
                    fmt.template_suffix = ntypemap.template_suffix
                else:
                    fmt.template_suffix = "_" + ntypemap.flat_name
            else:
                fmt.template_suffix = "_" + str(iargs)

            new.cxx_template = {}
            fmt.CXX_template = targs.instantiation  # ex. <int>

            # Gather headers required by template arguments.
            for targ in targs.asts:
                ntypemap = targ.typemap
                headers_typedef[ntypemap.name] = ntypemap

            self.push_instantiate_scope(new, targs)

            if new.ast.template_argument:
                iast = getattr(self.instantiate_scope, new.ast.template_argument)
                new.ast = new.ast.instantiate(node.ast.instantiate(iast))
                # Generics cannot differentiate on return type
                new.options.F_create_generic = False

            # Replace templated arguments.
            # arg - declast.Declaration
            newparams = []
            for arg in new.ast.declarator.params:
                if arg.template_argument:
                    iast = getattr(self.instantiate_scope, arg.template_argument)
                    newparams.append(arg.instantiate(iast))
                else:
                    newparams.append(arg)
            new.ast.declarator.params = newparams
            self.pop_instantiate_scope()

        new.gen_headers_typedef = headers_typedef
        # Do not process templated node, instead process
        # generated functions above.
        node.wrap.clear()

    def template_function2(self, node, ordered_functions):
        """ Create overloaded functions for each templated argument.

        - decl: template<typename T> class vector
          cxx_template:
          - instantiation: <int>
          - instantiation: <double>
          declarations:
          - decl: void push_back( const T& value+intent(in) );

        node.template_arguments = [ TemplateArgument('<int>'), TemplateArgument('<double>')]
                 TemplateArgument.asts[i].typemap

        Clone entire function then look for template arguments.
        Use when the function itself is not templated, but it has a templated argument
        from a class.
        function_suffix is not modified for functions in a templated class.
        Instead C_name_scope is used to distinguish the functions.

        Args:
            node -
            ordered_functions -
        """
        new = node.clone()
        ordered_functions.append(new)
        self.append_function_index(new)

        new._generated = "cxx_template"

        new.cxx_template = {}
        #        fmt.CXX_template = targs.instantiation   # ex. <int>

        #        self.push_instantiate_scope(new, targs)

        if new.ast.template_argument:
            iast = getattr(self.instantiate_scope, new.ast.template_argument)
            new.ast = new.ast.instantiate(node.ast.instantiate(iast))
            # Generics cannot differentiate on return type
            new.options.F_create_generic = False

        # Replace templated arguments.
        newparams = []
        for arg in new.ast.declarator.params:
            if arg.template_argument:
                iast = getattr(self.instantiate_scope, arg.template_argument)
                newparams.append(arg.instantiate(iast))
            else:
                newparams.append(arg)
        new.ast.declarator.params = newparams
        #        self.pop_instantiate_scope()

        # Do not process templated node, instead process
        # generated functions above.
        node.wrap.clear()

    def process_assumed_rank(self, node):
        """Convert assumed-rank argument into fortran_generic.

        At least one argument has assumed-rank.
        Create generic funcions for scalar plus each rank
        and set the rank for assumed-rank argument.
        Each argument with assumed-rank is give the same rank.

        This routine is not called with F_CFI since assumed-rank can
        be used directly without the need for generic functions.

        Parameters
        ----------
        node : ast.FunctionNode
        """
        # fortran_generic must already be empty
        options = node.options
        params = node.ast.declarator.params

        for rank in range(options.F_assumed_rank_min,
                          options.F_assumed_rank_max+1):
            newdecls = copy.deepcopy(params)
            for decl in newdecls:
                attrs = decl.declarator.attrs
                meta = decl.declarator.metaattrs
                if meta["assumed-rank"]:
                    # Replace dimension(..) with rank(n).
                    attrs["dimension"] = None
                    attrs["rank"] = rank
                    meta["assumed-rank"] = None
            generic = ast.FortranGeneric(
                "", function_suffix="_{}d".format(rank),
                decls=newdecls)
            node.fortran_generic.append(generic)

        # Remove assumed-rank from C function.
        for decl in params:
            attrs = decl.declarator.attrs
            meta = decl.declarator.metaattrs
            if meta["assumed-rank"]:
                attrs["dimension"] = None
                meta["assumed-rank"] = None
        node.declgen = node.ast.gen_decl()
        
    def generic_function(self, node, ordered_functions):
        """Create overloaded functions for each generic method.

        - decl: void GenericReal(double arg)
          fortran_generic:
          - decl: (float arg)
            function_suffix: float
          - decl: (double arg)
            function_suffix: double

        XXX - needs refinement.
        From generic.yaml
        - decl: void GetPointerAsPointer(
               void **addr+intent(out),
               int *type+intent(out)+hidden,
               size_t *size+intent(out)+hidden)
          fortran_generic:
          - decl: (float **addr+intent(out)+rank(1)+deref(pointer))
          - decl: (float **addr+intent(out)+rank(2)+deref(pointer))

        # scalar/array generic
          - decl: int SumValues(int *values, int nvalues)
          fortran_generic:
          - decl: (int *values)
          - decl: (int *values+rank(1))
        In this example, the original declaration of values is scalar.
        An additional bind(C) will be created to declare values as
        assumed-size argument since it has the rank(1) attribute.

        If all of the arguments are scalar native types, then several
        Fortran wrappers will be created to call the same C wrapper.
        The conversion of the arguments will be done by Fortran intrinsics.
        ex.  int(arg, kind=C_LONG)
        Otherwise, a C wrapper will be created for each Fortran function.
        arg_to_buff will be called after this which may create an additional
        C wrapper to deal with these arguments.

        Parameters
        ----------
        node : ast.FunctionNode
            Function with 'fortran_generic'
        ordered_functions : list
            Create functions are appended to this list.

        """
        for generic in node.fortran_generic:
            new = node.clone()
            ordered_functions.append(new)
            self.append_function_index(new)
            new._generated = "fortran_generic"
            fmt = new.fmtdict
            # XXX append to existing suffix
            if generic.fmtdict:
                fmt.update(generic.fmtdict)
            fmt.function_suffix = fmt.function_suffix + generic.function_suffix
            new.fortran_generic = {}
            new.wrap.assign(fortran=True)
            new.ast.declarator.params = generic.decls

            # Try to call original C function if possible.
            # All arguments are native scalar.
            need_wrapper = False
            if new.ast.declarator.is_indirect():
                need_wrapper = True
            
            for arg in new.ast.declarator.params:
                if arg.declarator.is_indirect():
                    need_wrapper = True
                    break
                elif arg.typemap.sgroup == "native":
                    pass
                else:
                    need_wrapper = True
                    break

            if need_wrapper:
                # The C wrapper is required to cast constants.
                # generic.yaml: GenericReal
                new.C_force_wrapper = True
                new.wrap.c = True
                new._PTR_C_CXX_index = node._function_index
            else:
                new._PTR_F_C_index = node._function_index
        
        # Do not process templated node, instead process
        # generated functions above.
        #        node.wrap.c = False
        node.wrap.fortran = False

    #        node.wrap.python = False

    def has_default_args(self, node, ordered_functions):
        """
        For each function which has a default argument, generate
        a version for each possible call.
          void func(int i = 0, int j = 0)
        generates
          void func()
          void func(int i)
          void func(int i, int j)
        In Fortran, these are added to a generic interface.

        Args:
            node -
            ordered_functions -
        """
        # Need to create individual routines for Fortran and C
        if node.wrap.fortran == False and node.wrap.c == False:
            return
        if node.options.F_default_args != "generic":
            return
        default_funcs = []

        default_arg_suffix = node.default_arg_suffix
        ndefault = 0

        min_args = 0
        for i, arg in enumerate(node.ast.declarator.params):
            if arg.declarator.init is None:
                min_args += 1
                continue
            new = node.clone()
            self.append_function_index(new)
            new._generated = "has_default_arg"
            del new.ast.declarator.params[i:]  # remove trailing arguments
            new._has_default_arg = False
            # Python and Lua both deal with default args in their own way
            new.wrap.assign(c=True, fortran=True)
            fmt = new.fmtdict
            try:
                fmt.function_suffix = default_arg_suffix[ndefault]
            except IndexError:
                # XXX fmt.function_suffix =
                # XXX  fmt.function_suffix + '_nargs%d' % (i + 1)
                pass
            default_funcs.append(new._function_index)
            ordered_functions.append(new)
            ndefault += 1

        # keep track of generated default value functions
        node._default_funcs = default_funcs
        node._nargs = (min_args, len(node.ast.declarator.params))
        # The last name calls with all arguments (the original decl)
        try:
            node.fmtdict.function_suffix = default_arg_suffix[ndefault]
        except IndexError:
            pass

    def move_arg_attributes(self, arg, old_node, new_node):
        """After new_node has been created from old_node,
        the result is being converted into an argument.
        Move some attributes that are associated with the function
        to the new argument.

        Note: Used with 'char *' and std::string arguments.

        Parameters
        ----------
        arg : ast.Declaration
            New argument, result of old_node.
        old_node : FunctionNode
            Original function (wrap fortran).
        new_node : FunctionNode
            New function (wrap c) that passes arg.
        """
        arg.metaattrs["deref"] = new_node.ast.metaattrs["deref"]
        new_node.ast.metaattrs["deref"] = None
            
        c_attrs = new_node.ast.attrs
        attrs = arg.attrs
        for name in ["owner", "free_pattern"]:
            if c_attrs[name]:
                attrs[name] = c_attrs[name]
                del c_attrs[name]

    def process_return_this(self, node, ordered_functions):
        """Deal with return_this feature.

        If a function is marked return_this, convert it into a 
        subroutine for the C and Fortran wrappers.
        Return this allows chaining of function calls.
        For example in C++:   obj->doA()->doB();
        Python:   obj.doA().doB()
        However, there is no way to chain in C or Fortran.

        Clone the function and wrap for C and Fortran.
        Turn off C and Fortran wrapper on original node.
        Remove the function result.
        
        Parameters
        ----------
        node : FunctionNode
        ordered_functions : list of FunctionNode
        """
        if node.wrap.c == False and node.wrap.fortran == False:
            return
        new = node.clone()
        ordered_functions.append(new)
        self.append_function_index(new)
        new._generated = "return_this"

        # Only wrap for C and Fortran, transfer values from node.
        new.wrap.clear()
        new.wrap.c = node.wrap.c
        new.wrap.fortran = node.wrap.fortran
        node.wrap.c = False
        node.wrap.fortran = False

        # Do not return C++ this instance.
        new.ast.set_return_to_void()
        new.ast.declarator.metaattrs["intent"] = "subroutine"
    
    def arg_to_CFI(self, node, ordered_functions):
        """Look for functions which can use TS29113
        Futher interoperability with C.

        If a function requires CFI_cdesc_t, clone the function and set
        arg.metaattrs["api"] to "cfi" to use the correct statements.  The
        new function will be called by Fortran directly via the
        bind(C) interface.  The original function no longer needs to
        be wrapped by Fortran; however, it will still be wrapped by C
        to provide a C API to a C++ function.
        

        Parameters
        ----------
        node : FunctionNode
        ordered_functions : list of FunctionNode
        """
        options = node.options
        fmt_func = node.fmtdict

        if options.wrap_fortran is False:
            # The buffer function is intended to be called by Fortran.
            # No Fortran, no need for buffer function.
            return

        ast = node.ast
        declarator = ast.declarator
        result_typemap = ast.typemap
        # shadow classes have not been added yet.
        # Only care about string, vector here.
        result_is_ptr = declarator.is_indirect()
        if (
            result_typemap
            and result_typemap.base in ["string", "vector"]
            and result_typemap.name != "char"
            and not result_is_ptr
        ):
            node.wrap.c = False
            #            node.wrap.fortran = False
            self.config.log.write(
                "Skipping {}, unable to create C wrapper "
                "for function returning {} instance"
                " (must return a pointer or reference)."
                " Bufferify version will still be created.\n".format(
                    result_typemap.cxx_type, declarator.user_name
                )
            )
        
        cfi_args = {}
        for arg in ast.declarator.params:
            declarator = arg.declarator
            name = declarator.user_name
            attrs = declarator.attrs
            meta = declarator.metaattrs
            cfi_args[name] = False
            arg_typemap = arg.typemap
            if meta["api"]:
                # API explicitly set by user.
                continue
            elif meta["assumed-rank"]:
                cfi_args[name] = True
            elif attrs["rank"]:
                cfi_args[name] = True
            elif arg_typemap.sgroup == "string":
                    cfi_args[name] = True
            elif arg_typemap.sgroup == "char":
                if declarator.is_indirect():
                    cfi_args[name] = True
            elif meta["deref"] in ["allocatable", "pointer"]:
                cfi_args[name] = True
        has_cfi_arg = any(cfi_args.values())

        # Function result.
        need_buf_result   = None

        result_as_arg = ""  # Only applies to string functions
        # when the result is added as an argument to the Fortran api.

        # Check if result needs to be an argument.
        declarator = ast.declarator
        attrs = declarator.attrs
        meta = declarator.metaattrs
        if meta["deref"] == "raw":
            # No bufferify required for raw pointer result.
            pass
        elif result_typemap.sgroup == "string":
            need_buf_result   = "cfi"
            result_as_arg = fmt_func.F_string_result_as_arg
            result_name = result_as_arg or fmt_func.C_string_result_as_arg
        elif result_typemap.sgroup == "char" and result_is_ptr:
            need_buf_result   = "cfi"
            result_as_arg = fmt_func.F_string_result_as_arg
            result_name = result_as_arg or fmt_func.C_string_result_as_arg
        elif meta["deref"] in ["allocatable", "pointer"]:
            need_buf_result   = "cfi"

        if not (need_buf_result or
                has_cfi_arg):
            return False

        # Create a new C function and change arguments
        # and add attributes.
        C_new = node.clone()
        ordered_functions.append(C_new)
        self.append_function_index(C_new)

        generated_suffix = "cfi"
        C_new._generated = "arg_to_cfi"
        C_new.splicer_group = "cfi"
        if need_buf_result:
            C_new.ast.declarator.metaattrs["api"] = need_buf_result
        if result_as_arg:
            C_new.ast.declarator.metaattrs["deref"] = "arg"
            # Special case for wrapf.py to override "allocatable"
            node.ast.declarator.metaattrs["deref"] = None
        fmt_func = C_new.fmtdict
        fmt_func.f_c_suffix = fmt_func.C_cfi_suffix

        C_new.wrap.assign(c=True, fortran=True)  # XXX - f_c
        C_new._PTR_C_CXX_index = node._function_index

        for arg in C_new.ast.declarator.params:
            name = arg.declarator.user_name
            if cfi_args[name]:
                arg.declarator.metaattrs["api"] = generated_suffix

        ast = C_new.ast

        node.wrap.fortran = False
        if node._generated in ["result_to_arg", "fortran_generic", "getter/setter"]:
            # Do not wrap intermediate functions
            node.wrap.c = False
        # Fortran function calls cfi/bufferify function.
        node._PTR_F_C_index = C_new._function_index
        return True

    def arg_to_buffer(self, node, ordered_functions):
        """Look for function which have buffer arguments.

        This includes functions with string or vector arguments.
        If found then create a new C function that
        sets metaattrs["api"] to 'buf'. This will find groups in
        fc_statements which will add arguments bufferify arguments
        (typically a buffer and length).

        String arguments added deref(allocatable) by default so that
        char * function will create an allocatable string in Fortran.

        Parameters
        ----------
        node : FunctionNode
        ordered_functions : list of FunctionNode
        """
        options = node.options
        fmt_func = node.fmtdict

        if node.wrap.c is False:
#        if options.wrap_c is False:  # XXX cdesc.yaml GetScalar2
            # The user does not require a C wrapper.
            # This can be the case if the Fortran wrapper is doing all
            # the work via splicer or fstatements.
            return

        # If a C++ function returns a std::string instance,
        # the default wrapper will not compile since the wrapper
        # will be declared as char. It will also want to return the
        # c_str of a stack variable. Warn and turn off the wrapper.
        ast = node.ast
        declarator = ast.declarator
        result_typemap = ast.typemap
        # shadow classes have not been added yet.
        # Only care about string, vector here.
        result_is_ptr = ast.declarator.is_indirect()
        if (
            result_typemap
            and result_typemap.base in ["string", "vector"]
            and result_typemap.name != "char"
            and not result_is_ptr
        ):
            node.wrap.c = False
            #            node.wrap.fortran = False
            self.config.log.write(
                "Skipping {}, unable to create C wrapper "
                "for function returning {} instance"
                " (must return a pointer or reference)."
                " Bufferify version will still be created.\n".format(
                    result_typemap.cxx_type, declarator.user_name
                )
            )

        if node.wrap.fortran is False:
            # The buffer function is intended to be called by Fortran.
            # No Fortran, no need for buffer function.
            return
        if options.F_string_len_trim is False:  # XXX what about vector?
            return

        # Arguments.
        # Is result or any argument a string or vector?
        # If so, additional arguments will be passed down so
        # create buffer version of function.
        buf_args = {}
        for arg in declarator.params:
            has_buf_arg = None
            arg_typemap = arg.typemap
            declarator = arg.declarator
            attrs = declarator.attrs
            meta = declarator.metaattrs
            if meta["api"]:
                # API explicitly set by user.
                pass
            elif arg_typemap.sgroup == "string":
                if meta["deref"] in ["allocatable", "pointer", "copy"]:
                    has_buf_arg = "cdesc"
                    # XXX - this is not tested
                    # XXX - tested with string **arg+intent(out)+dimension(ndim)
                else:
                    has_buf_arg = "buf"
            elif arg_typemap.sgroup == "char":
                if arg.ftrim_char_in:
                    pass
                elif declarator.is_indirect():
                    if meta["deref"] in ["allocatable", "pointer"]:
                        has_buf_arg = "cdesc"
                    else:
                        has_buf_arg = "buf"
            elif arg_typemap.sgroup == "vector":
                if meta["intent"] == "in":
                    # Pass SIZE.
                    has_buf_arg = "buf"
                else:
                    has_buf_arg = "cdesc"
            elif (arg_typemap.sgroup == "native" and
                  meta["intent"] == "out" and
                  meta["deref"] != "raw" and
                  declarator.get_indirect_stmt() in ["**", "*&"]):
                # double **values +intent(out) +deref(pointer)
                has_buf_arg = "cdesc"
                #has_buf_arg = "buf" # XXX - for scalar?
            buf_args[declarator.user_name] = has_buf_arg
        # --- End loop over function parameters
        has_buf_arg = any(buf_args.values())

        # Function result.
        need_buf_result   = None

        result_as_arg = ""  # Only applies to string functions
        # when the result is added as an argument to the Fortran api.

        # Check if result needs to be an argument.
        attrs = ast.declarator.attrs
        meta = ast.declarator.metaattrs
        if meta["deref"] == "raw":
            # No bufferify required for raw pointer result.
            pass
        elif result_typemap.sgroup == "string":
            if meta["deref"] in ["allocatable", "pointer"]:
                need_buf_result = "cdesc"
            else:
                need_buf_result = "buf"
            result_as_arg = fmt_func.F_string_result_as_arg
            result_name = result_as_arg or fmt_func.C_string_result_as_arg
        elif result_typemap.sgroup == "char" and result_is_ptr:
            if meta["deref"] in ["allocatable", "pointer"]:
                # Result default to "allocatable".
                need_buf_result = "cdesc"
            else:
                need_buf_result = "buf"
            result_as_arg = fmt_func.F_string_result_as_arg
            result_name = result_as_arg or fmt_func.C_string_result_as_arg
        elif result_typemap.base == "vector":
            need_buf_result = "cdesc"
        elif result_is_ptr:
            if meta["deref"] in ["allocatable", "pointer"]:
                if meta["dimension"]:
                    # int *get_array() +deref(pointer)+dimension(10)
                    need_buf_result = "cdesc"

        # Functions with these results need wrappers.
        if not (need_buf_result or
                has_buf_arg):
            return

        # XXX       node.wrap.fortran = False
        # Preserve wrap.c.
        # This keep a version which accepts char * arguments.

        # Create a new C function and change arguments
        # and add attributes.
        C_new = node.clone()
        ordered_functions.append(C_new)
        self.append_function_index(C_new)

        generated_suffix = "buf"
        C_new._generated = "arg_to_buffer"
        C_new.splicer_group = "buf"
        if need_buf_result:
            C_new.ast.declarator.metaattrs["api"] = need_buf_result
        if result_as_arg:
            C_new.ast.declarator.metaattrs["deref"] = "arg"
            # Special case for wrapf.py to override "allocatable"
            node.ast.declarator.metaattrs["deref"] = None
            # XXX - the legacy wrapper uses buf, easier to call from C.
            C_new.ast.declarator.metaattrs["api"] = "buf"
        
        fmt_func = C_new.fmtdict
        fmt_func.f_c_suffix = fmt_func.C_bufferify_suffix

        options = C_new.options
        C_new.wrap.assign(c=True, fortran=True)  # XXX - f_c
        C_new._PTR_C_CXX_index = node._function_index

        for arg in C_new.ast.declarator.params:
            declarator = arg.declarator
            attrs = declarator.attrs
            meta = declarator.metaattrs
            if buf_args[declarator.user_name]:
                meta["api"] = buf_args[declarator.user_name]
            if arg.ftrim_char_in:
                continue
            arg_typemap = arg.typemap
            if arg_typemap.base == "vector":
                # Do not wrap the orignal C function with vector argument.
                # Meaningless to call without the size argument.
                # TODO: add an option where char** length is determined by looking
                #       for trailing NULL pointer.  { "foo", "bar", NULL };
                node.wrap.c = False
                node.wrap.lua = False  # NotImplemented

        node.wrap.fortran = False
        if node._generated in ["result_to_arg", "fortran_generic", "getter/setter"]:
            node.wrap.c = False
            
        # Fortran function may call C subroutine if string/vector result
        node._PTR_F_C_index = C_new._function_index

    def XXXcheck_class_dependencies(self, node):
        """
        Check used_types and find which header and module files
        to use for this class
        """
        # keep track of types which are used by methods arguments
        used_types = {}
        for method in node["methods"]:
            self.check_function_dependencies(method, used_types)

        modules = {}
        for typ in used_types.values():
            if typ.f_module:
                for mname, only in typ.f_module.items():
                    module = modules.setdefault(mname, {})
                    if only:  # Empty list means no ONLY clause
                        for oname in only:
                            module[oname] = True

        # Always add C_PTR, needed for class F_derived_member
        modules.setdefault("iso_c_binding", {})["C_PTR"] = True

        F_modules = []  # array of tuples ( name, (only1, only2) )
        for mname in sorted(modules):
            F_modules.append((mname, sorted(modules[mname])))
        node.F_module_dependencies = F_modules

    def XXXcheck_function_dependencies(self, node, used_types):
        """Record which types are used by a function.
        """
        if node.cxx_template:
            # The templated type will raise an error.
            # XXX - Maybe dummy it out
            # XXX - process templated types
            return
        ast = node.ast
        result_typemap = ast.typemap
        # XXX - make sure it exists
        used_types[result_typemap.name] = result_typemap
        for arg in ast.declarator.params:
            ntypemap = arg.typemap
            if ntypemap.base == "shadow":
                used_types[ntypemap.name] = ntypemap

    def gen_functions_decl(self, functions):
        """ Generate declgen for all functions.

        Args:
            functions - list of ast.FunctionNode.
        """
        for node in functions:
            node.declgen = node.ast.gen_decl()


class Namify(object):
    """Compute names of functions in library.
    Need to compute F_name and F_C_name since they interact.
    Compute all C names first, then Fortran.
    A Fortran function may call a generated C function via
    _PTR_F_C_index
    Also compute number which may be controlled by options.

    C_name - Name of C function
    F_C_name - Fortran function for C interface
    F_name_impl - Name of Fortran function implementation
    """

    def __init__(self, newlibrary, config):
        """
        Args:
            newlibrary - ast.LibraryNode
            config -
        """
        self.newlibrary = newlibrary
        self.config = config

    def name_library(self):
        """entry pointer for library"""
        self.name_language(self.newlibrary.wrap_namespace)
        self.name_language(self.newlibrary.wrap_namespace)

    def iter_decl(self, node):
        """Loop over members of a Namespace, class"""
        for func in node.functions:
            func.update_names()
        
    def name_language(self, node):
        """
        Args:
            handler - function.
            node - ast.LibraryNode, ast.NamespaceNode
        """
        for cls in node.classes:
            self.iter_decl(cls)

        self.iter_decl(node)

        for ns in node.namespaces:
            self.name_language(ns)


class Preprocess(object):
    """Compute some state for functions."""

    def __init__(self, newlibrary, config):
        """
        Args:
            newlibrary - ast.LibraryNode
            config -
        """
        self.newlibrary = newlibrary
        self.config = config

    def process_library(self):
        """entry pointer for library"""
        self.process_namespace(self.newlibrary.wrap_namespace)

    def process_namespace(self, node):
        """Process a namespace.

        Args:
            node - ast.LibraryNode, ast.NamespaceNode
        """
        for cls in node.classes:
            for func in cls.functions:
                self.process_function(cls, func)

        for func in node.functions:
            self.process_function(None, func)

        for ns in node.namespaces:
            self.process_namespace(ns)

    def process_function(self, cls, node):
        """
         Args:
            cls -
            node -
        """
        attrs = node.ast.declarator.attrs
        meta = node.ast.declarator.metaattrs


def generate_functions(library, config):
    whelpers.set_library(library)
    VerifyAttrs(library, config).verify_attrs()
    GenFunctions(library, config).gen_library()
    Namify(library, config).name_library()
    Preprocess(library, config).process_library()
    ast.promote_wrap(library)

######################################################################

class TemplateTypemap(visitor.Visitor):
    """Visit nodes in AST.

    Can be used as a base class to traverse AST.
    """
    def __init__(self, config):
        # config may be None in unitttests.
        self.config = config
        super(TemplateTypemap, self).__init__()

    def visit_LibraryNode(self, node):
        for cls in node.classes:
            self.visit(cls)
        for fcn in node.functions:
            self.visit(fcn)
        for ns in node.namespaces:
            self.visit(ns)
        for var in node.variables:
            self.visit(var)

    def visit_ClassNode(self, node):
        for cls in node.classes:
            self.visit(cls)
        for fcn in node.functions:
            if fcn.ast.declarator.is_ctor():
                fcn.ast.typemap = node.typemap
            self.visit(fcn)
        for var in node.variables:
            self.visit(var)

    visit_NamespaceNode = visit_LibraryNode

    def visit_FunctionNode(self, node):
        self.visit(node.ast)

    def visit_VariableNode(self, node):
        pass

    def visit_Declaration(self, ast):
        """Find template typemap

        1) Get template arguments as a string into targs (ex. <int>)
        2) Look up in the typemap assigned by the parser,
           typically the original class. 
        3) Replace typemap in AST.

        Complain if the AST template has not been instantiated.
        """
        if ast.template_arguments and ast.typemap.cxx_instantiation is not None:
            targs = ast.gen_template_arguments()
            template_typemap = ast.typemap.cxx_instantiation.get(targs, None)
            if template_typemap is None:
                if self.config:
                    self.config.log.write(
                        "ERROR: Template {}{} is not instantiated\n"
                        .format(ast.typemap.cxx_type, targs))
            else:
                ast.typemap = template_typemap
        
        if ast.declarator.params is not None:
            for arg in ast.declarator.params:
                self.visit(arg)

######################################################################


class CheckImplied(todict.PrintNode):
    """Check arguments in the implied attribute.
    """

    def __init__(self, context, expr, decls):
        super(CheckImplied, self).__init__()
        self.context = context
        self.expr = expr
        self.decls = decls

    def visit_Identifier(self, node):
        """Check arguments to implied attribute.

        Args:
            node -
        """
        if node.args is None:
            # Not a function.
            return node.name
        elif node.name == "size":
            # size(arg)
            if len(node.args) > 2:
                error.get_cursor().generate(
                    "Too many arguments to 'size': {}".format(self.expr)
                )
            # isinstance(node.args[0], declalst.Identifier)
            argname = node.args[0].name
            arg = declast.find_arg_by_name(self.decls, argname)
            if arg is None:
                error.get_cursor().generate(
                    "Unknown argument '{}': {}".format(argname, self.expr)
                )
            return "size"
        elif node.name in ["len", "len_trim"]:
            # len(arg)  len_trim(arg)
            if len(node.args) != 1:
                error.get_cursor().generate(
                    "Too many arguments to '{}': {}".format(node.name, self.expr)
                )
            argname = node.args[0].name
            arg = declast.find_arg_by_name(self.decls, argname)
            if arg is None:
                error.get_cursor().generate(
                    "Unknown argument '{}': {}".format(argname, self.expr)
                )
            # XXX - Make sure character
#            if arg.attrs["dimension"] is None:
#                raise RuntimeError(
#                    "Argument '{}' must have dimension attribute: {}".format(
#                        argname, self.expr
#                    )
#                )
            return node.name
        else:
            # Assume a user defined function.
            return self.param_list(node)


def check_implied_attrs(context, decls):
    """Check all parameters for implied arguments.

    The implied attribute may reference other arguments in decls.
    Only call on the full Fortran decls.
    If fortran_generic, call for each decls member.
    Otherwise, call on FunctionNode.ast.declarator.params

    Args:
        context  - contains node.linenumber
        decls - list of Declarations
    """
    for decl in decls:
        expr = decl.declarator.attrs["implied"]
        if expr:
            check_implied(context, expr, decls)


def check_implied(context, expr, decls):
    """Check implied attribute expression for errors.
    expr may reference other arguments in decls.

    Args:
        expr  - implied attribute value
        decls - list of Declarations
    """
    node = declast.ExprParser(expr).expression()
    visitor = CheckImplied(context, expr, decls)
    return visitor.visit(node)
