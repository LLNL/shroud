# Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and
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

    def verify_attrs(self):
        """Verify library attributes.
        Recurse through all declarations.
        Entry pointer for class VerifyAttrs.
        """
        self.verify_namespace_attrs(self.newlibrary.wrap_namespace)

    def verify_namespace_attrs(self, node):
        """Verify attributes for a library or namespace.

        Args:
            node - ast.LibraryNode, ast.NameSpaceNode
        """
        for cls in node.classes:
            for var in cls.variables:
                self.check_var_attrs(cls, var)
            for func in cls.functions:
                self.check_fcn_attrs(func)

        for func in node.functions:
            self.check_fcn_attrs(func)

        for ns in node.namespaces:
            self.verify_namespace_attrs(ns)

    def check_var_attrs(self, cls, node):
        """Check attributes for variables.
        This includes struct and class members.

        Args:
            cls -
            node -
        """
        ast = node.ast
        attrs = ast.attrs
        for attr in attrs:
            if attr[0] == "_":  # internal attribute
                continue
            if attr not in ["name", "readonly", "dimension"]:
                raise RuntimeError(
                    "Illegal attribute '{}' for variable '{}' at line {}".format(
                        attr, ast.name, node.linenumber
                    ) + "\nonly 'name', 'readonly', and 'dimension' are allowed on variables"
                )

        is_ptr = ast.is_indirect()
        if attrs["dimension"] and not is_ptr:
            raise RuntimeError(
                "dimension attribute can only be "
                "used on pointer and references"
            )

        self.parse_attrs(node, ast)

    def check_fcn_attrs(self, node):
        """Check attributes on FunctionNode.

        Args:
            node - ast.FunctionNode
        """
        options = node.options

        ast = node.ast
        node._has_found_default = False

        for attr in ast.attrs:
            if attr[0] == "_":  # internal attribute
                continue
            if attr not in [
                "api",          # arguments to pass to C wrapper.
                "allocatable",  # return a Fortran ALLOCATABLE
                "cdesc",
                "deref",  # How to dereference pointer
                "dimension",
                "free_pattern",
                "len",
                "name",
                "owner",
                "pure",
                "rank",
            ]:
                raise RuntimeError(
                    "Illegal attribute '{}' for function '{}' define at line {}".format(
                        attr, node.ast.name, node.linenumber
                    )
                )
        if ast.typemap is None:
            print("XXXXXX typemap is None")
        if ast.typemap.sgroup == "shadow":
            if options.C_shadow_result:
                ast.metaattrs["api"] = "capptr"
            else:
                ast.metaattrs["api"] = "capsule"
        if ast.is_ctor():
            ast.metaattrs["intent"] = "ctor"
        elif ast.is_dtor():
            ast.metaattrs["intent"] = "dtor"
        else:
            ast.metaattrs["intent"] = ast.get_subprogram()
        self.check_deref_attr_func(node)
        self.check_common_attrs(node.ast)

        for arg in ast.params:
            if arg.declarator is None:
                raise RuntimeError("Argument must have name in {} at line {}".format(
                    node.decl, node.linenumber))
            self.check_arg_attrs(node, arg)

        if node.fortran_generic:
            for generic in node.fortran_generic:
                for garg in generic.decls:
                    generic._has_found_default = False
                    self.check_arg_attrs(generic, garg, node.options)
                check_implied_attrs(node, generic.decls)
        else:
            check_implied_attrs(node, ast.params)

        self.parse_attrs(node, ast)

    def check_intent_attr(self, node, arg):
        """Set default intent meta-attribute.

        Intent is only valid on arguments.
        intent: lower case, no parens, must be in, out, or inout
        """
        attrs = arg.attrs
        meta = arg.metaattrs
        is_ptr = arg.is_indirect()
        intent = arg.attrs["intent"]
        if intent is None:
            if node is None:
                # do not default intent for function pointers
                pass
            elif arg.is_function_pointer():
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
            if intent in ["in", "out", "inout"]:
                meta["intent"] = intent
            else:
                raise RuntimeError("Bad value for intent: " + attrs["intent"])
            if not is_ptr and intent != "in":
                # Nonpointers can only be intent(in).
                raise RuntimeError("{}: Only pointer arguments may have intent attribute".format(node.linenumber))
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
        attrs = ast.attrs
        deref = attrs["deref"]
        mderef = None
        ntypemap = ast.typemap
        nindirect = ast.is_indirect()
        if ast.get_subprogram() == "subroutine":
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
                raise RuntimeError(
                    "Cannot have attribute 'deref' on function which returns multiple indirections in {}".
                    format(node.decl))
        elif nindirect == 1:
            # pointer to a POD  e.g. int *
            if deref:
                mderef = deref
            elif ntypemap.sgroup == "char":  # char *
                if attrs["len"]:
                    mderef = "copy"
                else:
                    mderef = "allocatable"
            elif attrs["dimension"]:
                mderef = "pointer"
            else:
                mderef = options.return_scalar_pointer
        elif deref:
            raise RuntimeError(
                "Cannot have attribute 'deref' on non-pointer in {}".format(
                    node.decl
                )
            )
        ast.metaattrs["deref"] = mderef
        
    def check_deref_attr_var(self, node, ast):
        """Check deref attr and set default for variable.

        Pointer variables set the default deref meta attribute.

        Parameters
        ----------
        node - ast.FunctionNode or ast.FortranGeneric
        ast : declast.Declaration
        """
        attrs = ast.attrs
        meta = ast.metaattrs
        ntypemap = ast.typemap
        is_ptr = ast.is_indirect()

        deref = attrs["deref"]
        if deref is not None:
            if deref not in ["allocatable", "pointer", "raw", "scalar"]:
                raise RuntimeError(
                    "Illegal value '{}' for deref attribute. "
                    "Must be 'allocatable', 'pointer', 'raw', "
                    "or 'scalar'.".format(deref)
                )
            nindirect = ast.is_indirect()
            if ntypemap.sgroup == "vector":
                if deref:
                    mderef = deref
                else:
                    # Copy vector to new array.
                    mderef = "allocatable"
            elif nindirect != 2:
                raise RuntimeError(
                    "Can only have attribute 'deref' on arguments which"
                    " return a pointer:"
                    " '{}' at line {}".format(ast.name, node.linenumber))
            elif meta["intent"] == "in":
                raise RuntimeError(
                    "Cannot have attribute 'deref' on intent(in) argument:"
                    " '{}' at line".format(ast.name, node.linenumber))
            meta["deref"] = attrs["deref"]
            return

        # Set deref attribute for arguments which return values.
        intent = meta["intent"]
        spointer = ast.get_indirect_stmt()
        if ntypemap.name == "void":
            # void cannot be dereferenced.
            pass
        elif spointer in ["**", "*&"] and intent == "out":
            meta["deref"] = "pointer"
            
    def check_common_attrs(self, ast):
        """Check attributes which are common to function and argument AST
        This includes: dimension, free_pattern, owner, rank

        Parameters
        ----------
        ast : declast.Declaration
        """
        attrs = ast.attrs
        meta = ast.metaattrs
        ntypemap = ast.typemap
        is_ptr = ast.is_indirect()

        # api
        api = attrs["api"]
        if api is None:
            pass
        elif api not in ["capi", "buf", "cfi"]:
                raise RuntimeError(
                    "'api' attribute must 'capi', 'buf', or 'cfi'"
                )
        else:
            meta["api"] = api

        # dimension
        dimension = attrs["dimension"]
        rank = attrs["rank"]
        if rank:
            if rank is True:
                raise RuntimeError(
                    "'rank' attribute must have an integer value"
                )
            try:
                attrs["rank"] = int(attrs["rank"])
            except ValueError:
                raise RuntimeError(
                    "'rank' attribute must have an integer value, not '{}'"
                    .format(attrs["rank"])
                )
            if attrs["rank"] > 7:
                raise RuntimeError(
                    "'rank' attribute must be 0-7, not '{}'"
                    .format(attrs["rank"])
                )
            if not is_ptr:
                raise RuntimeError(
                    "rank attribute can only be "
                    "used on pointer and references"
                )
        if dimension:
            if dimension is True:
                raise RuntimeError(
                    "dimension attribute must have a value."
                )
            if attrs["value"]:
                raise RuntimeError(
                    "argument may not have 'value' and 'dimension' attribute."
                )
            if rank:
                raise RuntimeError(
                    "argument may not have 'rank' and 'dimension' attribute."
                )
            if not is_ptr:
                raise RuntimeError(
                    "dimension attribute can only be "
                    "used on pointer and references"
                )
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
                raise RuntimeError(
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
        if options is None:
            options = node.options
        argname = arg.name
        attrs = arg.attrs
        meta = arg.metaattrs

        for attr in attrs:
            if attr[0] == "_":  # Shroud internal attribute.
                continue
            if attr not in [
                "api",
                "allocatable",
                "assumedtype",
                "blanknull",   # Treat blank string as NULL pointer.
                "capsule",
                "cdesc",
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
                raise RuntimeError(
                    "Illegal attribute '{}' for argument '{}' defined at line {}".format(
                        attr, argname, node.linenumber
                    )
                )

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

        is_ptr = arg.is_indirect()

        # assumedtype
        assumedtype = attrs["assumedtype"]
        if assumedtype is not None:
            if attrs["value"]:
                raise RuntimeError(
                    "argument must not have value=True "
                    "because it has the assumedtype attribute."
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
                raise RuntimeError(
                    "charlen attribute can only be "
                    "used on 'char *'"
                )
            if is_ptr != 1:
                raise RuntimeError(
                    "charlen attribute can only be "
                    "used on 'char *'"
                )
            if charlen is True:
                raise RuntimeError("charlen attribute must have a value")

        char_ptr_in = (
            is_ptr == 1 and
            intent == "in" and
            arg_typemap.name == "char")
            
        blanknull = attrs["blanknull"]
        if blanknull is not None:
            if not char_ptr_in:
                raise RuntimeError(
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
            if arg.init is not None:
                node._has_default_arg = True
            elif node._has_found_default is True:
                raise RuntimeError("Expected default value for %s" % argname)

        # Check template attribute
        temp = arg.template_arguments
        if arg_typemap and arg_typemap.base == "vector":
            if not temp:
                raise RuntimeError(
                    "line {}: std::vector must have template argument: {}".format(
                        node.linenumber, arg.gen_decl()
                    )
                )
            arg_typemap = arg.template_arguments[0].typemap
            if arg_typemap is None:
                raise RuntimeError(
                    "check_arg_attr: No such type %s for template: %s"
                    % (temp, arg.gen_decl())
                )
        elif temp:
            raise RuntimeError(
                "Type '%s' may not supply template argument: %s"
                % (arg_typemap.name, arg.gen_decl())
            )

        self.parse_attrs(node, arg)

        # Flag node if any argument is assumed-rank.
        if arg.metaattrs["assumed-rank"]:
            node._gen_fortran_generic = True

        if arg.is_function_pointer():
            for arg1 in arg.params:
                self.check_arg_attrs(None, arg1, options)

    def parse_attrs(self, node, ast):
        """Parse dimension attributes and save the AST.
        This tree will be traversed by the wrapping classes
        to convert to language specific code.

        Parameters
        ----------
        node : ast.FunctionNode, ast.FortranGeneric, ast.VariableNode
            Used for line number in error messages.
        ast : declast.Declaration
        """
        attrs = ast.attrs
        metaattrs = ast.metaattrs

        dim = attrs["dimension"]
        if dim:
            try:
                check_dimension(dim, metaattrs)
            except RuntimeError:
                raise RuntimeError("Unable to parse dimension: {} at line {}"
                                   .format(dim, node.linenumber))


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
              define_result_as_arg_functions
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

    def gen_library(self):
        """Entry routine to generate functions for a library.
        """
        newlibrary = self.newlibrary
        whelpers.add_all_helpers()

        self.function_index = newlibrary.function_index
        self.class_map = newlibrary.class_map

        self.instantiate_all_classes(newlibrary.wrap_namespace)
        self.update_templated_typemaps(newlibrary.wrap_namespace)
        self.gen_namespace(newlibrary.wrap_namespace)

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
            scope = getattr(node, "scope", "")  # XXX - ClassNode has scope
            setattr(newscope, scope + node.template_parameters[idx], argast)
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
            fattrs = node.ast.attrs
            found_ctor = found_ctor or fattrs["_constructor"]
            found_dtor = found_dtor or fattrs["_destructor"]

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
        sgroup = ast.typemap.sgroup

        fmt = util.Scope(var.fmtdict)
        fmt_func = dict(
            # Use variable's field_name for the generated functions.
            field_name=var.fmtdict.field_name, # Name of member variable
            wrapped_name=ast.name,             # Using name attribute
            struct_name=cls.fmtdict.cxx_type,
        )

        is_struct = cls.wrap_as == "struct"
        if is_struct:
            if not options.F_struct_getter_setter:
                return
            if var.ast.is_pointer() != 1:
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
        elif var.ast.is_pointer():
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
        meta = fcn.ast.metaattrs
        meta.update(ast.metaattrs)
        meta["intent"] = "getter"
        meta["deref"] = deref
        if is_struct:
            meta["struct"] = cls.typemap.flat_name
            meta = fcn.ast.params[0].metaattrs
            meta["struct"] = cls.typemap.flat_name
            meta["intent"] = "in"
        fcn.wrap.lua = False
        fcn.wrap.python = False
        fcn._generated = "getter/setter"

        ##########
        # setter
        if ast.attrs["readonly"]:
            return
        argdecl = ast.gen_arg_as_language(lang=lang, name="val", continuation=True)
        decl = "void {}({}{})".format(funcname_set, this_set, argdecl)

        attrs = dict(
            val=dict(
                intent="in",
                value=value,
            )
        )
        dim = ast.metaattrs["dimension"]
        if dim:
            attrs["val"]["rank"] = len(dim)

        fcn = parent.add_function(decl, attrs=attrs, format=fmt_func)
        # XXX - The function is not processed like other, so set intent directly.
        fcn.ast.metaattrs["intent"] = "setter"
        iarg = 0
        if is_struct:
            meta = fcn.ast.params[0].metaattrs
            meta["intent"] = "inout"
            meta["struct"] = cls.typemap.flat_name
            iarg = 1
        meta = fcn.ast.params[iarg].metaattrs
        meta.update(ast.metaattrs)
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
                    #  cxx_class - vector_0 or vector_int     (Fortran and C names)
                    #  cxx_type  - vector<int>
                    cxx_class = "{}{}".format(
                        newcls.fmtdict.cxx_class, class_suffix
                    )
                    cxx_type = "{}{}".format(
                        newcls.fmtdict.cxx_class, targs.instantiation
                    )

                    if targs.fmtdict and "cxx_class" in targs.fmtdict:
                        # Check if user has changed cxx_class.
                        cxx_class = targs.fmtdict["cxx_class"]

                    newcls.scope_file[-1] += class_suffix
                    # Add default values to format dictionary.
                    newcls.fmtdict.update(
                        dict(
                            cxx_type=cxx_type,
                            cxx_class=cxx_class,
                            class_scope=cxx_class + "::",
                            C_name_scope=newcls.parent.fmtdict.C_name_scope + cxx_class + "_",
                            F_name_scope=newcls.parent.fmtdict.F_name_scope + cxx_class.lower() + "_",
                            F_derived_name=cxx_class.lower(),
                            file_scope="_".join(newcls.scope_file[1:]),
                        )
                    )
                    # Add user specified values to format dictionary.
                    if targs.fmtdict:
                        newcls.fmtdict.update(targs.fmtdict)

                    # Remove defaulted attributes, load files from fmtdict, recompute defaults
                    newcls.delete_format_templates()

                    # Update format and options from template_arguments
                    if targs.fmtdict:
                        newcls.fmtdict.update(targs.fmtdict)
                    if targs.options:
                        newcls.options.update(targs.options)

                    newcls.expand_format_templates()
                    newcls.typemap = typemap.create_class_typemap(newcls)
                    if targs.instantiation in orig_typemap.cxx_instantiation:
                        print("instantiate_classes: {} already in "
                              "typemap.cxx_instantiation".format(targs.instantiation))
                    orig_typemap.cxx_instantiation[targs.instantiation] = newcls.typemap

                    self.push_instantiate_scope(newcls, targs)
                    self.process_class(newcls, newcls)
                    self.pop_instantiate_scope()
            else:
                clslist.append(cls)
                self.process_class(cls, cls)

        node.classes = clslist

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
        name = ast.attrs["name"]  #cls.name + "_ctor"
        #  Add variables as function parameters by coping AST.
        for var in cls.variables:
            a = copy.deepcopy(var.ast)
            a.metaattrs["intent"] = "in"
            a.metaattrs["struct_member"] = var
            ast.params.append(a)
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
            raise RuntimeError("process_class: class already exists in class_map")
        self.class_map[cls.typemap.flat_name] = cls
        for var in cls.variables:
            self.add_var_getter_setter(parent, cls, var)
        cls.functions = self.define_function_suffix(cls.functions)

    def define_result_as_arg_functions(self, functions):
        """Convert result into argument if requested."""
        ordered = []
        for node in functions:
            ordered.append(node)
            if node.wrap.fortran is False:
                continue
            self.convert_result_as_arg(node, ordered)
        return ordered

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
            cxx_overload.setdefault(function.ast.name, []).append(
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
            if function.ast.is_ctor():
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
                overloaded_functions.setdefault(function.ast.name, []).append(
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
                
        ordered3a = self.define_result_as_arg_functions(ordered2)
        ordered3 = self.define_fortran_generic_functions(ordered3a)
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

            if new.ast.typemap.base == "template":
                iast = getattr(self.instantiate_scope, new.ast.typemap.name)
                new.ast = new.ast.instantiate(node.ast.instantiate(iast))
                # Generics cannot differentiate on return type
                new.options.F_create_generic = False

            # Replace templated arguments.
            # arg - declast.Declaration
            newparams = []
            for arg in new.ast.params:
                if arg.typemap.base == "template":
                    iast = getattr(self.instantiate_scope, arg.typemap.name)
                    newparams.append(arg.instantiate(iast))
                else:
                    newparams.append(arg)
            new.ast.params = newparams
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

        if new.ast.typemap.base == "template":
            iast = getattr(self.instantiate_scope, new.ast.typemap.name)
            new.ast = new.ast.instantiate(node.ast.instantiate(iast))
            # Generics cannot differentiate on return type
            new.options.F_create_generic = False

        # Replace templated arguments.
        newparams = []
        for arg in new.ast.params:
            if arg.typemap.base == "template":
                iast = getattr(self.instantiate_scope, arg.typemap.name)
                newparams.append(arg.instantiate(iast))
            else:
                newparams.append(arg)
        new.ast.params = newparams
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
        params = node.ast.params

        for rank in range(options.F_assumed_rank_min,
                          options.F_assumed_rank_max+1):
            newdecls = copy.deepcopy(params)
            for decl in newdecls:
                if decl.metaattrs["assumed-rank"]:
                    # Replace dimension(..) with rank(n).
                    decl.attrs["dimension"] = None
                    decl.attrs["rank"] = rank
                    decl.metaattrs["assumed-rank"] = None
            generic = ast.FortranGeneric(
                "", function_suffix="_{}d".format(rank),
                decls=newdecls)
            node.fortran_generic.append(generic)

        # Remove assumed-rank from C function.
        for decl in params:
            if decl.metaattrs["assumed-rank"]:
                decl.attrs["dimension"] = None
                decl.metaattrs["assumed-rank"] = None
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
            new.ast.params = generic.decls

            # Try to call original C function if possible.
            # All arguments are native scalar.
            need_wrapper = False
            if new.ast.is_indirect():
                need_wrapper = True
            
            for arg in new.ast.params:
                if arg.is_indirect():
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
        for i, arg in enumerate(node.ast.params):
            if arg.init is None:
                min_args += 1
                continue
            new = node.clone()
            self.append_function_index(new)
            new._generated = "has_default_arg"
            del new.ast.params[i:]  # remove trailing arguments
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
        node._nargs = (min_args, len(node.ast.params))
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

    def result_as_arg(self, node, C_new):
        """
        Create a Fortran function for a C function which has the
        result added as an argument.

        Create Fortran function without bufferify function_suffix but
        with len attributes on string arguments.
        char *out(); ->  call out(result_as_arg)
        """
        F_new = C_new.clone()

        # Fortran function should wrap the new C function
        F_new._PTR_F_C_index = C_new._function_index
        F_new.wrap.assign(fortran=True)
        # Do not add '_bufferify'
        F_new.fmtdict.function_suffix = node.fmtdict.function_suffix

        # Do not wrap original function (does not have result argument)
        node.wrap.fortran = False
        return F_new

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
        new.ast.metaattrs["intent"] = "subroutine"
    
    def convert_result_as_arg(self, node, ordered_functions):
        """Convert a function result into an argument.

        Used when the YAML file names the output argument
        via fmt.F_string_result_as_arg for (char or string)
        Create a new function.

        For example, a function returns a 'char *' write a Fortran
        wrapper which accepts a CHARACTER(*) argument which will be
        assigned the function result.  This helps memory management
        issues at the cost of copying data.

        The original function will still be wrapped in C but returns a
        type(C_PTR) where the user must call c_f_pointer themselves.
        """
        return ordered_functions # XXX - do nothing for now
        options = node.options
        fmt_func = node.fmtdict
#        if options.F_string_len_trim is False:  # XXX what about vector?
#            return

        ast = node.ast
        result_typemap = ast.typemap
        result_name = None

        # Check if result needs to be an argument.
        attrs = ast.attrs
        meta = ast.metaattrs
        if meta["deref"] == "raw":
            # No bufferify required for raw pointer result.
            pass
        elif result_typemap.sgroup in ["char", "string"]:
            result_name = fmt_func.F_string_result_as_arg
#            result_as_arg = fmt_func.F_string_result_as_arg
#            result_name = result_as_arg or fmt_func.C_string_result_as_arg
#        elif result_typemap.base == "vector":
#            has_vector_result = True
#        elif result_is_ptr:
#            if meta["deref"] in ["allocatable", "pointer"]:
#                need_cdesc_result = True
#            elif attrs["dimension"]:
#                need_cdesc_result = True

        if not result_name:
            return

##########
        # Create a new C function and change arguments
        # and add attributes.
        C_new = node.clone()
        ordered_functions.append(C_new)
        self.append_function_index(C_new)

#        generated_suffix = "buf"
        C_new._generated = "result_to_arg"
        fmt_func = C_new.fmtdict
#        fmt_func.function_suffix = fmt_func.function_suffix + fmt_func.C_bufferify_suffix + "XXX"
#        fmt_func.function_suffix = fmt_func.function_suffix

        options = C_new.options
        C_new.wrap.assign(c=True, fortran=True)
        C_new._PTR_C_CXX_index = node._function_index
##########

        # decl: const char * getCharPtr2()
        new_arg = C_new.ast.result_as_arg(result_name)
        new_arg.const = False # must be writeable
#        attrs = new_arg.attrs
#        new_arg.metaattrs["deref"] = None
        # Special case for wrapf.py to override "allocatable"

        # Special case for wrapf.py to override "allocatable"
        node.ast.metaattrs["deref"] = None
        new_arg.metaattrs["deref"] = "result"
        new_arg.metaattrs["is_result"] = True
        C_new.ast.metaattrs["intent"] = "subroutine"
        C_new.ast.metaattrs["deref"] = None

        node.wrap.fortran = False
#        node.wrap.c = False

        return
        F_new = self.result_as_arg(node, C_new)
        ordered_functions.append(F_new)
        self.append_function_index(F_new)

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
        result_typemap = ast.typemap
        # shadow classes have not been added yet.
        # Only care about string, vector here.
        result_is_ptr = ast.is_indirect()
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
                    result_typemap.cxx_type, ast.name
                )
            )
        
        ast = node.ast
        cfi_args = {}
        for arg in ast.params:
            cfi_args[arg.name] = False
            arg_typemap = arg.typemap
            if arg.metaattrs["api"]:
                # API explicitly set by user.
                continue
            elif arg.metaattrs["assumed-rank"]:
                cfi_args[arg.name] = True
            elif arg.attrs["rank"]:
                cfi_args[arg.name] = True
            elif arg_typemap.sgroup == "string":
                    cfi_args[arg.name] = True
            elif arg_typemap.sgroup == "char":
                if arg.is_indirect():
                    cfi_args[arg.name] = True
            elif arg.metaattrs["deref"] in ["allocatable", "pointer"]:
                cfi_args[arg.name] = True
        has_cfi_arg = any(cfi_args.values())

        # Function result.
        need_buf_result   = None

        result_as_arg = ""  # Only applies to string functions
        # when the result is added as an argument to the Fortran api.

        # Check if result needs to be an argument.
        attrs = ast.attrs
        meta = ast.metaattrs
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

        options.wrap_fortran = False

        # Create a new C function and change arguments
        # and add attributes.
        C_new = node.clone()
        ordered_functions.append(C_new)
        self.append_function_index(C_new)

        generated_suffix = "cfi"
        C_new._generated = "arg_to_cfi"
        C_new.splicer_group = "cfi"
        if need_buf_result:
            C_new.ast.metaattrs["api"] = need_buf_result
        fmt_func = C_new.fmtdict
        fmt_func.function_suffix = fmt_func.function_suffix + fmt_func.C_cfi_suffix

        C_new.wrap.assign(c=True)#, fortran=True)
        C_new._PTR_C_CXX_index = node._function_index

        for arg in C_new.ast.params:
            if cfi_args[arg.name]:
                arg.metaattrs["api"] = generated_suffix
            attrs = arg.attrs
            arg_typemap = arg.typemap

        ast = C_new.ast
        if True: # preserve to avoid changing indention for now.
            f_attrs = node.ast.attrs  # Fortran function attributes
            f_meta = node.ast.metaattrs  # Fortran function attributes
            if result_as_arg:
                # decl: const char * getCharPtr2() +len(30)
                # +len implies copying into users buffer.
                result_as_string = ast.result_as_arg(result_name)
                result_as_string.const = False # must be writeable
                attrs = result_as_string.attrs
                # Special case for wrapf.py to override "allocatable"
                f_meta["deref"] = None
                result_as_string.metaattrs["api"] = "cfi"
                result_as_string.metaattrs["deref"] = "result"
                result_as_string.metaattrs["is_result"] = True
                C_new.ast.metaattrs["api"] = None
                C_new.ast.metaattrs["intent"] = "subroutine"
                C_new.ast.metaattrs["deref"] = None

        if result_as_arg:
            F_new = self.result_as_arg(node, C_new)
            ordered_functions.append(F_new)
            self.append_function_index(F_new)
        else:
            if node._generated in ["result_to_arg", "fortran_generic", "getter/setter"]:
                node.wrap.c = False
            # Fortran function may call C subroutine if string/vector result
            # Fortran function calls bufferify function.
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
        result_typemap = ast.typemap
        # shadow classes have not been added yet.
        # Only care about string, vector here.
        result_is_ptr = ast.is_indirect()
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
                    result_typemap.cxx_type, ast.name
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
        for arg in ast.params:
            has_buf_arg = None
            arg_typemap = arg.typemap
            if arg.metaattrs["api"]:
                # API explicitly set by user.
                continue
            elif arg.attrs["cdesc"]:
                # User requested cdesc.
                has_buf_arg = "cdesc"
            elif arg_typemap.sgroup == "string":
                if arg.metaattrs["deref"] in ["allocatable", "pointer"]:
                    has_buf_arg = "cdesc"
                    # XXX - this is not tested
                else:
                    has_buf_arg = "buf"
            elif arg_typemap.sgroup == "char":
                if arg.ftrim_char_in:
                    pass
                elif arg.is_indirect():
                    if arg.metaattrs["deref"] in ["allocatable", "pointer"]:
                        has_buf_arg = "cdesc"
                    else:
                        has_buf_arg = "buf"
            elif arg_typemap.sgroup == "vector":
                if arg.metaattrs["intent"] == "in":
                    # Pass SIZE.
                    has_buf_arg = "buf"
                else:
                    has_buf_arg = "cdesc"
            elif (arg_typemap.sgroup == "native" and
                  arg.metaattrs["intent"] == "out" and
                  arg.metaattrs["deref"] != "raw" and
                  arg.get_indirect_stmt() in ["**", "*&"]):
                # double **values +intent(out) +deref(pointer)
                has_buf_arg = "cdesc"
                #has_buf_arg = "buf" # XXX - for scalar?
            buf_args[arg.name] = has_buf_arg
        has_buf_arg = any(buf_args.values())

        # Function result.
        need_buf_result   = None

        result_as_arg = ""  # Only applies to string functions
        # when the result is added as an argument to the Fortran api.

        # Check if result needs to be an argument.
        attrs = ast.attrs
        meta = ast.metaattrs
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
                else:
                    # int *get_scalar() +deref(pointer)
                    need_buf_result = "buf"

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
            C_new.ast.metaattrs["api"] = need_buf_result
        
        fmt_func = C_new.fmtdict
        fmt_func.function_suffix = fmt_func.function_suffix + fmt_func.C_bufferify_suffix

        options = C_new.options
        C_new.wrap.assign(c=node.options.wrap_c)
        C_new._PTR_C_CXX_index = node._function_index

        for arg in C_new.ast.params:
            if buf_args[arg.name]:
                arg.metaattrs["api"] = buf_args[arg.name]
            attrs = arg.attrs
            meta = arg.metaattrs
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

        ast = C_new.ast
        if True: # preserve to avoid changing indention for now.
            # Add additional argument to hold result.
            # This will allocate a new character variable to hold the
            # results of the C++ function.
            f_attrs = node.ast.attrs  # Fortran function attributes
            f_meta = node.ast.metaattrs  # Fortran function attributes

            if result_as_arg:
                # decl: const char * getCharPtr2() +len(30)
                # +len implies copying into users buffer.
                result_as_string = ast.result_as_arg(result_name)
                result_as_string.const = False # must be writeable
                attrs = result_as_string.attrs
                # Special case for wrapf.py to override "allocatable"
                f_meta["deref"] = None
                # We've added an argument to fill, use api=buf.
                result_as_string.metaattrs["api"] = "buf"
                result_as_string.metaattrs["deref"] = "result"
                result_as_string.metaattrs["is_result"] = True
                C_new.ast.metaattrs["api"] = None
                C_new.ast.metaattrs["intent"] = "subroutine"
                C_new.ast.metaattrs["deref"] = None

        if result_as_arg:
            F_new = self.result_as_arg(node, C_new)
            ordered_functions.append(F_new)
            self.append_function_index(F_new)
        else:
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
        for arg in ast.params:
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
        self.name_language(self.name_function_c, self.newlibrary.wrap_namespace)
        self.name_language(self.name_function_fortran, self.newlibrary.wrap_namespace)

    def name_language(self, handler, node):
        """
        Args:
            handler - function.
            node - ast.LibraryNode, ast.NamespaceNode
        """
        for cls in node.classes:
            for func in cls.functions:
                handler(func)

        for func in node.functions:
            handler(func)

        for ns in node.namespaces:
            self.name_language(handler, ns)

    def name_function_c(self, node):
        """Compute name for C wrapped functions.

        Parameters
        ----------
        node : FunctionNode
        """
        if not node.wrap.c:
            return
        fmt_func = node.fmtdict

        node.eval_template("C_name")
        node.eval_template("F_C_name")
        fmt_func.F_C_name = fmt_func.F_C_name.lower()

    def name_function_fortran(self, node):
        """Compute name for Fortran wrapped functions.

        Parameters
        ----------
        node : FunctionNode
        """
        if not node.wrap.fortran:
            return

        node.eval_template("F_name_impl")
        node.eval_template("F_name_function")
        node.eval_template("F_name_generic")


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
        attrs = node.ast.attrs
        meta = node.ast.metaattrs
        if attrs["owner"] == "caller" and \
           meta["deref"] == "pointer":
            meta["capsule"] = True

#        for arg in node.ast.params:
#   XXX - check for capsule on stuff like 'int **var +intent(out)+dimension(10)'


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
            if fcn.ast.is_ctor():
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
        
        if ast.params is not None:
            for arg in ast.params:
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
                raise RuntimeError(
                    "{}:Too many arguments to 'size': {}".format(
                        self.context.linenumber, self.expr)
                )
            # isinstance(node.args[0], declalst.Identifier)
            argname = node.args[0].name
            arg = declast.find_arg_by_name(self.decls, argname)
            if arg is None:
                raise RuntimeError(
                    "{}:Unknown argument '{}': {}".format(
                        self.context.linenumber, argname, self.expr)
                )
            return "size"
        elif node.name in ["len", "len_trim"]:
            # len(arg)  len_trim(arg)
            if len(node.args) != 1:
                raise RuntimeError(
                    "{}:Too many arguments to '{}': {}".format(
                        self.context.linenumber, node.name, self.expr)
                )
            argname = node.args[0].name
            arg = declast.find_arg_by_name(self.decls, argname)
            if arg is None:
                raise RuntimeError(
                    "{}:Unknown argument '{}': {}".format(
                        self.context.linenumber, argname, self.expr)
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
    Otherwise, call on FunctionNode.ast.params

    Args:
        context  - contains node.linenumber
        decls - list of Declarations
    """
    for decl in decls:
        expr = decl.attrs["implied"]
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

