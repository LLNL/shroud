# Copyright (c) 2017-2020, Lawrence Livermore National Security, LLC and
# other Shroud Project Developers.
# See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

"""
Generate additional functions required to create wrappers.
"""
from __future__ import print_function
from __future__ import absolute_import

import copy

from . import ast
from . import declast
from . import todict
from . import typemap
from . import util
from . import whelpers

wformat = util.wformat


class VerifyAttrs(object):
    """
    Check attributes and set some defaults.
    Generate types for classes.
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
                    )
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

        for attr in node.ast.attrs:
            if attr[0] == "_":  # internal attribute
                continue
            if attr not in [
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
        self.check_shared_attrs(node.ast)

        if ast.typemap is None:
            print("XXXXXX typemap is None")
        for arg in ast.params:
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
            
    def check_shared_attrs(self, ast):
        """Check attributes which may be assigned to function or argument:
        deref, dimension, free_pattern, owner, rank

        Args:
            ast - declast.Declaration
        """
        attrs = ast.attrs
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
        # XXX deref only on pointer, vector

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
        intent: lower case, no parens, must be in, out, or inout
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

        for attr in attrs:
            if attr[0] == "_":  # Shroud internal attribute.
                continue
            if attr not in [
                "allocatable",
                "assumedtype",
                "capsule",
                "cdesc",
                "charlen",
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

        self.check_shared_attrs(arg)

        is_ptr = arg.is_indirect()

        # intent
        intent = attrs["intent"]
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
            elif arg_typemap.sgroup == "void":
                # void *
                intent = "in"  # XXX must coordinate with VALUE
            else:
                intent = "inout"
            attrs["intent"] = intent
            # XXX - Do hidden arguments need intent?
        else:
            intent = intent.lower()
            if intent in ["in", "out", "inout"]:
                attrs["intent"] = intent
            else:
                raise RuntimeError("Bad value for intent: " + attrs["intent"])
            if not is_ptr and intent != "in":
                # Nonpointers can only be intent(in).
                raise RuntimeError("{}: Only pointer arguments may have intent attribute".format(node.linenumber))

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

        # Set deref attribute for arguments which return values.
        spointer = arg.get_indirect_stmt()
        if attrs["deref"]:
            # User defined.
            pass
        elif arg_typemap.name == "void":
            # void cannot be dereferenced.
            pass
        elif spointer in ["**", "*&"] and intent == "out":
            attrs["deref"] = "pointer"
                
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

        if intent == "in" and is_ptr == 1 and arg_typemap.name == "char":
            # const char *arg
            # char *arg+intent(in)
            arg.ftrim_char_in = True

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

        if arg.is_function_pointer():
            for arg1 in arg.params:
                self.check_arg_attrs(None, arg1, options)

    def parse_attrs(self, node, ast):
        """Parse dimension attributes and save the AST.
        This tree will be traversed by the wrapping classes
        to convert to language specific code.

        Args:
            attrs - collections.defaultdict
            node - Container struct or class.
        """
        attrs = ast.attrs
        metaattrs = ast.metaattrs
        
        dim = attrs["dimension"]
        if dim:
            try:
                metaattrs["dimension"] = declast.check_dimension(dim)
            except RuntimeError:
                raise RuntimeError("Unable to parse dimension: {} at line {}"
                                   .format(dim, node.linenumber))


class GenFunctions(object):
    """
    Generate Typemap from class.
    Generate functions based on overload/template/generic/attributes
    Computes fmt.function_suffix.
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

    def gen_library(self):
        """Entry routine to generate functions for a library.
        """
        newlibrary = self.newlibrary
        whelpers.add_all_helpers()

        self.function_index = newlibrary.function_index

        self.instantiate_all_classes(newlibrary.wrap_namespace)
        self.gen_namespace(newlibrary.wrap_namespace)

    def gen_namespace(self, node):
        """Process functions which are not in classes

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
            targs - list of TemplateArguments
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

    def add_var_getter_setter(self, cls, var):
        """Create getter/setter functions for class variables.
        This allows wrappers to access class members.

        Do not wrap for Python since descriptors are created for
        class member variables.

        Args:
            cls -
            var -
        """
        ast = var.ast
        arg_typemap = ast.typemap
        fieldname = ast.name  # attrs["name"]

        fmt = util.Scope(var.fmtdict)

        options = dict(wrap_lua=False, wrap_python=False)

        # getter
        funcname = "get" + fieldname.capitalize()
        argdecl = ast.gen_arg_as_c(name=funcname, continuation=True)
        decl = "{}()".format(argdecl)
        field = wformat("{CXX_this}->{field_name}", fmt)
        if arg_typemap.cxx_to_c is None:
            val = field
        else:
            fmt.cxx_var = field
            val = wformat(arg_typemap.cxx_to_c, fmt)

        splicer = dict(
            c=[
                "return " + val + ";",
            ],
        )

        cls.add_function(decl, options=options, splicer=splicer)

        # setter
        if ast.attrs["readonly"]:
            return
        funcname = "set" + ast.name.capitalize()
        argdecl = ast.gen_arg_as_c(name="val", continuation=True)
        decl = "void {}({})".format(funcname, argdecl)
        field = wformat("{CXX_this}->{field_name}", fmt)
        if arg_typemap.c_to_cxx is None:
            val = "val"
        else:
            fmt.c_var = "val"
            val = wformat(arg_typemap.c_to_cxx, fmt)
        set_val = "{} = {};".format(field, val)

        attrs = dict(
            val=dict(
                intent="in", value=True
            )  # XXX - what about pointer variables?
        )

        splicer = dict(
            c=[
                set_val,
                "return;"
            ],
        )

        cls.add_function(decl, attrs=attrs, options=options, splicer=splicer)

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
        node - LibraryNode or ClassNode.

        Create a new list of classes replacing
        any class with template_arguments with instantiated classes.
        All new classes will be added to node.classes.

        Args:
            node - ast.LibraryNode, ast.NamespaceNode, ast.ClassNode
        """
        clslist = []
        for cls in node.classes:
            if cls.as_struct:
                clslist.append(cls)
                options = cls.options
                if options.wrap_python and options.PY_struct_arg == "class":
                    self.add_struct_ctor(cls)
            elif cls.template_arguments:
                # Replace class with new class for each template instantiation.
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
                    self.update_types_for_class_instantiation(newcls)

                    self.push_instantiate_scope(newcls, targs)
                    self.process_class(newcls)
                    self.pop_instantiate_scope()
            else:
                clslist.append(cls)
                self.process_class(cls)

        node.classes = clslist

    def add_struct_ctor(self, cls):
        """Add a constructor function for a struct when
        it will be treated like a class.

        Args:
            cls - ast.ClassNode
        """
        ast = declast.create_struct_ctor(cls)
        name = ast.attrs["name"]  #cls.name + "_ctor"
        #  Add variables as function parameters by coping AST.
        for var in cls.variables:
            a = copy.deepcopy(var.ast)
            a.attrs["intent"] = "in"
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

    def update_types_for_class_instantiation(self, cls):
        """Update the references to use instantiated class.

        Args:
            cls - ast.ClassNode
        """
        for function in cls.functions:
            if function.ast.is_ctor():
                function.ast.typemap = cls.typemap

    def process_class(self, cls):
        """process variables and functions for a class.
        Create getter/setter functions for class variables.

        Args:
            cls - ast.ClassNode
        """
        for var in cls.variables:
            self.add_var_getter_setter(cls, var)
        cls.functions = self.define_function_suffix(cls.functions)

    def define_function_suffix(self, functions):
        """
        Return a new list with generated function inserted.

        Args:
            functions - list of ast.FunctionNode
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
            # if not function.options.wrap_c:
            #     continue
            if function.cxx_template:
                continue
            if function.template_arguments:
                continue
            if function.have_template_args:
                # Stuff like push_back which is in a templated class, is not an overload
                # C_name_scope is used to distigunish the functions, not function_suffix.
                continue
            if function.ast.is_ctor():
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

        # Create additional C bufferify functions.
        ordered3 = []
        for method in ordered_functions:
            ordered3.append(method)
            if method.options.F_create_bufferify_function:
                self.arg_to_buffer(method, ordered3)

        # Create multiple generic Fortran wrappers to call a
        # single C functions
        ordered4 = []
        for method in ordered3:
            ordered4.append(method)
            if not method.options.wrap_fortran:
                continue
            if method.fortran_generic:
                method._overloaded = True
                self.generic_function(method, ordered4)

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
        headers_typedef = {}

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
            options = new.options
            options.wrap_c = oldoptions.wrap_c
            options.wrap_fortran = oldoptions.wrap_fortran
            options.wrap_python = oldoptions.wrap_python
            options.wrap_lua = oldoptions.wrap_lua
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
                options.F_create_generic = False

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
        options = node.options
        options.wrap_c = False
        options.wrap_fortran = False
        options.wrap_python = False
        options.wrap_lua = False

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
        oldoptions = node.options

        new = node.clone()
        ordered_functions.append(new)
        self.append_function_index(new)

        new._generated = "cxx_template"

        new.cxx_template = {}
        options = new.options
        options.wrap_c = oldoptions.wrap_c
        options.wrap_fortran = oldoptions.wrap_fortran
        options.wrap_python = oldoptions.wrap_python
        options.wrap_lua = oldoptions.wrap_lua
        #        fmt.CXX_template = targs.instantiation   # ex. <int>

        #        self.push_instantiate_scope(new, targs)

        if new.ast.typemap.base == "template":
            iast = getattr(self.instantiate_scope, new.ast.typemap.name)
            new.ast = new.ast.instantiate(node.ast.instantiate(iast))
            # Generics cannot differentiate on return type
            options.F_create_generic = False

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
        options = node.options
        options.wrap_c = False
        options.wrap_fortran = False
        options.wrap_python = False
        options.wrap_lua = False

    def generic_function(self, node, ordered_functions):
        """ Create overloaded functions for each generic method.

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
        The C wrapper must pass down a context argument to allow
        the shape information to be returned. Normally, this would be
        added by arg_to_buffer, but since the C argument is 'void **'
        it will not be.  But the generic function does have
        an argument which meets the critieria.

        Args:
            node - ast.FunctionNode
            ordered_functions -
        """
        context_args = {}
        for generic in node.fortran_generic:
            new = node.clone()
            ordered_functions.append(new)
            self.append_function_index(new)
            new._generated = "fortran_generic"
            new._PTR_F_C_index = node._function_index
            fmt = new.fmtdict
            # XXX append to existing suffix
            if generic.fmtdict:
                fmt.update(generic.fmtdict)
            fmt.function_suffix = fmt.function_suffix + generic.function_suffix
            new.fortran_generic = {}
            options = new.options
            options.wrap_c = False
            options.wrap_fortran = True
            options.wrap_python = False
            options.wrap_lua = False
            new.ast.params = generic.decls

            for arg in generic.decls:
                # double **arg +intent(out)+rank(1)
                if (arg.typemap.sgroup == "native" and
                    arg.attrs["intent"] == "out" and
                    arg.get_indirect_stmt() in  ["**", "*&"]):
                    context_args[arg.name] = True
                
        for argname in context_args.keys():
            arg = node.ast.find_arg_by_name(argname)
            arg.attrs["context"] = "FIXME"

        # Do not process templated node, instead process
        # generated functions above.
        options = node.options
        #        options.wrap_c = False
        options.wrap_fortran = False

    #        options.wrap_python = False

    def has_default_args(self, node, ordered_functions):
        """
        For each function which has a default argument, generate
        a version for each possible call.
          void func(int i = 0, int j = 0)
        generates
          void func()
          void func(int i)
          void func(int i, int j)

        Args:
            node -
            ordered_functions -
        """
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
            options = new.options
            options.wrap_c = True
            options.wrap_fortran = True
            # Python and Lua both deal with default args in their own way
            options.wrap_python = False
            options.wrap_lua = False
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

    def move_arg_attributes(self, attrs, old_node, new_node):
        """After new_node has been created from old_node,
        the result is being converted into an argument.
        Move some attributes that are associated with the function
        to the new argument.

        If deref is not set, then default to allocatable.

        Args:
            attrs - attributes of the new argument
            old_node - The FunctionNode of the original function.
            new_node - The FunctionNode of the new function with
        """
        c_attrs = new_node.ast.attrs
        f_attrs = old_node.ast.attrs
        if f_attrs["deref"] is None:
            f_attrs["deref"] = "allocatable"
            attrs["deref"] = "allocatable"
        for name in ["owner", "free_pattern"]:
            if c_attrs[name]:
                attrs[name] = c_attrs[name]
                del c_attrs[name]

    def arg_to_buffer(self, node, ordered_functions):
        """Look for function which have buffer arguments.
        This includes functions with string or vector arguments.
        If found then create a new C function that
        will add arguments buf_args (typically a buffer and length).

        String arguments added deref(allocatable) by default so that
        char * function will create an allocatable string in Fortran.

        Args:
            node -
            ordered_functions -
        """
        options = node.options
        fmt = node.fmtdict

        if options.wrap_c is False:
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
        attrs = ast.attrs
        result_is_ptr = ast.is_indirect()
        if (
            result_typemap
            and result_typemap.base in ["string", "vector"]
            and result_typemap.name != "char"
            and not result_is_ptr
        ):
            options.wrap_c = False
            #            options.wrap_fortran = False
            self.config.log.write(
                "Skipping {}, unable to create C wrapper "
                "for function returning {} instance"
                " (must return a pointer or reference)."
                " Bufferify version will still be created.\n".format(
                    result_typemap.cxx_type, ast.name
                )
            )

        if options.wrap_fortran is False:
            # The buffer function is intended to be called by Fortran.
            # No Fortran, no need for buffer function.
            return
        if options.F_string_len_trim is False:  # XXX what about vector?
            return

        # Arguments.
        # Is result or any argument a string or vector?
        # If so, additional arguments will be passed down so
        # create buffer version of function.
        has_buf_arg = False
        for arg in ast.params:
            arg_typemap = arg.typemap
            if arg_typemap.base == "string":
                if arg.ftrim_char_in:
                    pass
                elif arg.is_indirect():
                    has_buf_arg = True
            elif arg_typemap.base == "vector":
                has_buf_arg = True
            elif (arg_typemap.sgroup == "native" and
                  arg.attrs["intent"] == "out" and
                  arg.get_indirect_stmt() in ["**", "*&"]):
#                 arg.attrs["dimension"]:
                # double **values +intent(out) +dimension(nvalues)
                has_buf_arg = True

        # Function Result.
        has_string_result = False
        has_vector_result = False
        need_cdesc_result = False

        result_as_arg = ""  # Only applies to string functions
        # when the result is added as an argument to the Fortran api.

        if attrs["deref"] == "raw":
            # No bufferify required for raw pointer result.
            pass
        elif result_typemap.base == "string":
            has_string_result = True
            result_as_arg = fmt.F_string_result_as_arg
            result_name = result_as_arg or fmt.C_string_result_as_arg
        elif result_typemap.base == "vector":
            has_vector_result = True
        elif result_is_ptr:
            if attrs["deref"] in ["allocatable"]:
                need_cdesc_result = True
            elif attrs["dimension"]:
                need_cdesc_result = True

        # Functions with these results need wrappers.
        if not (has_string_result or has_vector_result or
                need_cdesc_result or has_buf_arg):
            return

        # XXX       options = node['options']
        # XXX       options.wrap_fortran = False
        # Preserve wrap_c.
        # This keep a version which accepts char * arguments.

        # Create a new C function and change arguments
        # and add attributes.
        C_new = node.clone()
        ordered_functions.append(C_new)
        self.append_function_index(C_new)

        generated_suffix = "buf"
        C_new._generated = "arg_to_buffer"
        C_new.generated_suffix = generated_suffix  # used to lookup fc_statements
        fmt = C_new.fmtdict
        fmt.function_suffix = fmt.function_suffix + fmt.C_bufferify_suffix

        options = C_new.options
        options.wrap_c = True
        options.wrap_fortran = False
        options.wrap_python = False
        options.wrap_lua = False
        C_new._PTR_C_CXX_index = node._function_index

        for arg in C_new.ast.params:
            attrs = arg.attrs
            if arg.ftrim_char_in:
                continue
            arg_typemap = arg.typemap
            sgroup = arg_typemap.sgroup
            specialize = ""
            if arg_typemap.base == "vector":
                # Do not wrap the orignal C function with vector argument.
                # Meaningless to call without the size argument.
                # TODO: add an option where char** length is determined by looking
                #       for trailing NULL pointer.  { "foo", "bar", NULL };
                node.options.wrap_c = False
                node.options.wrap_lua = False  # NotImplemented
                specialize = arg.template_arguments[0].typemap.sgroup
            elif (sgroup == "native" and
                  arg.attrs["intent"] == "out" and
                  arg.get_indirect_stmt() in ["**", "*&"]):
#                 arg.attrs["dimension"]:
                attrs["context"] = True
            arg_typemap, sp = typemap.lookup_c_statements(arg)

            # Set names for implied buffer arguments.
            # This filters out "buf" for ftrim_char_in
            arg.stmts_suffix = generated_suffix
            
            spointer = arg.get_indirect_stmt()
            c_stmts = ["c", sgroup, spointer, attrs["intent"], generated_suffix, specialize]
            intent_blk = typemap.lookup_fc_stmts(c_stmts)
            typemap.create_buf_variable_names(options, intent_blk, attrs)

        ast = C_new.ast
        if has_string_result:
            # Add additional argument to hold result.
            # Default to deref(allocatable).
            # This will allocate a new character variable to hold the
            # results of the C++ function.
            f_attrs = node.ast.attrs  # Fortran function attributes

            if ast.attrs["len"] or result_as_arg:
                # +len implies copying into users buffer.
                result_as_string = ast.result_as_arg(result_name)
                result_as_string.const = False # must be writeable
                attrs = result_as_string.attrs
                attrs["len"] = options.C_var_len_template.format(
                    c_var=result_name
                )
                # Special case for wrapf.py to override "allocatable"
                f_attrs["deref"] = "result-as-arg"
            elif (result_typemap.cxx_type == "std::string" or
                  result_is_ptr):  # 'char *'
                result_as_string = ast.result_as_arg(result_name)
                attrs = result_as_string.attrs
                attrs["context"] = options.C_var_context_template.format(
                    c_var=result_name
                )
                self.move_arg_attributes(attrs, node, C_new)
            else:  # char
                result_as_string = ast.result_as_arg(result_name)
                result_as_string.const = False # must be writeable
                attrs = result_as_string.attrs
                attrs["len"] = options.C_var_len_template.format(
                    c_var=result_name
                )
            attrs["intent"] = "out"
            attrs["_is_result"] = True
            # convert to subroutine
            C_new._subprogram = "subroutine"
        elif has_vector_result:
            # Pass an argument to C wrapper for the function result.
            # XXX - string_result -> vector_result -> result
            vector_as_arg = fmt.F_string_result_as_arg
            result_name = vector_as_arg or fmt.C_string_result_as_arg
            result_as_vector = ast.result_as_arg(result_name)
            attrs = result_as_vector.attrs
            attrs["context"] = options.C_var_context_template.format(
                c_var=result_name
            )
            self.move_arg_attributes(attrs, node, C_new)
            attrs["intent"] = "out"
            attrs["_is_result"] = True
            # convert to subroutine
            C_new._subprogram = "subroutine"
        elif need_cdesc_result:
            # Non-string and Non-char results
            # XXX - c_var is duplicated in wrapc.py wrap_function
            fmt_func = C_new.fmtdict
            attrs = C_new.ast.attrs
            result_name = fmt_func.C_local + fmt_func.C_result
            attrs["context"] = options.C_var_context_template.format(
                c_var=result_name)

        if result_as_arg:
            # Create Fortran function without bufferify function_suffix but
            # with len attributes on string arguments.
            #  char *out(); ->  call out(result_as_arg)
            F_new = C_new.clone()
            ordered_functions.append(F_new)
            self.append_function_index(F_new)

            # Fortran function should wrap the new C function
            F_new._PTR_F_C_index = C_new._function_index
            options = F_new.options
            options.wrap_c = False
            options.wrap_fortran = True
            options.wrap_python = False
            options.wrap_lua = False
            # Do not add '_bufferify'
            F_new.fmtdict.function_suffix = node.fmtdict.function_suffix

            # Do not wrap original function (does not have result argument)
            node.options.wrap_fortran = False
        else:
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
                handler(cls, func)

        for func in node.functions:
            handler(None, func)

        for ns in node.namespaces:
            self.name_language(handler, ns)

    def name_function_c(self, cls, node):
        """
        Args:
            cls -
            node -
        """
        options = node.options
        if not options.wrap_c:
            return
        fmt_func = node.fmtdict

        node.eval_template("C_name")
        node.eval_template("F_C_name")
        fmt_func.F_C_name = fmt_func.F_C_name.lower()

    def name_function_fortran(self, cls, node):
        """ Must process C functions to generate their names.

        Args:
            cls -
            node -
        """
        options = node.options
        if not options.wrap_fortran:
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
        # Any nodes with cxx_template have been replaced with nodes
        # that have the template expanded.
        if not node.cxx_template:
            self.process_xxx(cls, node)
            self.check_return_pointer(node, node.ast)

        options = self.newlibrary.options
        # XXX - not sure if result uses any of these attributes.
#        typemap.set_buf_variable_names(
#            options, node.ast.attrs, "aaa")

        attrs = node.ast.attrs
        if attrs["owner"] == "caller" and \
           attrs["deref"] == "pointer" and \
           attrs["capsule"] is None:
            attrs["capsule"] = options.C_var_capsule_template.format(
                c_var=node.fmtdict.C_result
            )

        for arg in node.ast.params:
            typemap.set_buf_variable_names(
                options, arg.attrs, arg.name)

    def process_xxx(self, cls, node):
        """Compute information common to all wrapper languages.

        Compute subprogram.  This may be different for each language.
        CXX_subprogram - The C++ function being wrapped.
        C_subprogram - functions will be converted to subroutines for
            return_this and destructors.
            A subroutine can be converted to a function by C_return_type.

        return_this = True for C++ functions which return 'this',
        are easier to call from Fortran if they are subroutines.
        There is no way to chain in Fortran:  obj->doA()->doB();

#        Lookup up typemap for result and arguments

        Args:
            cls -
            node -
        """

        fmt_func = node.fmtdict

        ast = node.ast
        CXX_result_type = ast.typemap.name
        C_result_type = CXX_result_type
        F_result_type = CXX_result_type
        subprogram = ast.get_subprogram()
        node.CXX_subprogram = subprogram
        is_dtor = ast.attrs["_destructor"]

        if node.return_this or is_dtor:
            CXX_result_type = "void"
            C_result_type = "void"
            F_result_type = "void"
            node.CXX_subprogram = "subroutine"
            subprogram = "subroutine"

        node.C_subprogram = subprogram
        node.F_subprogram = subprogram

        node.CXX_return_type = CXX_result_type
        node.C_return_type = C_result_type
        node.F_return_type = F_result_type

        node.CXX_result_typemap = typemap.lookup_type(CXX_result_type)
        node.C_result_typemap = typemap.lookup_type(C_result_type)
        node.F_result_typemap = typemap.lookup_type(F_result_type)

    #        if not result_typedef:
    #            raise RuntimeError("Unknown type {} in {}",
    #                               CXX_result_type, fmt_func.function_name)

    def check_return_pointer(self, node, ast):
        """Compute how to deal with a pointer function result.

        Args:
            node - ast.FunctionNode
            ast - declast.Declaration
        """
        options = node.options
        attrs = ast.attrs
        result_typemap = node.CXX_result_typemap
        if result_typemap.cxx_type == "void":
            # subprogram == subroutine
            # deref may be set when a string function is converted into a subroutine.
            pass
        elif result_typemap.base == "shadow":
            # Change a C++ pointer into a Fortran pointer
            # return 'void *' as 'type(C_PTR)'
            # 'shadow' assigns pointer to type(C_PTR) in a derived type
            pass
        elif result_typemap.base in ["string", "vector"]:
            if attrs["deref"]:
                pass
            else:
                # Default strings to create a Fortran allocatable.
                # XXX - do not deref a scalar.
                if ast.is_indirect():
                    attrs["deref"] = "allocatable"
        elif ast.is_indirect():
            # pointer to a POD  e.g. int *
            if attrs["deref"]:
                pass
            elif attrs["dimension"]:
                attrs["deref"] = "pointer"
            else:
                attrs["deref"] = options.return_scalar_pointer
        else:
            if attrs["deref"]:
                raise RuntimeError(
                    "Cannot have attribute 'deref' on non-pointer in {}".format(
                        node.decl
                    )
                )


def generate_functions(library, config):
    whelpers.set_library(library)
    VerifyAttrs(library, config).verify_attrs()
    GenFunctions(library, config).gen_library()
    Namify(library, config).name_library()
    Preprocess(library, config).process_library()


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
            if len(node.args) != 1:
                raise RuntimeError(
                    "{}:Too many arguments to 'size': ".format(
                        self.context.linenumber, self.expr)
                )
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
