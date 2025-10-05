# Copyright Shroud Project Developers. See LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

"""
Check attributes in delcaration.
Generate additional functions required to create wrappers.
"""
from __future__ import print_function
from __future__ import absolute_import

import collections
import copy

from . import ast
from . import declast
from . import declstr
from . import error
from . import todict
from . import statements
from . import typemap
from . import util
from . import visitor

wformat = util.wformat


class VerifyAttrs(object):
    """
    Check attributes and set some defaults in metaattrs.
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
        node._has_found_default = False

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
        arg_typemap = arg.typemap

        # charlen
        # Only meaningful with 'char *arg+intent(out)'
        # XXX - Python needs a value if 'char *+intent(out)'
        charlen = attrs.get("charlen", None)
        if charlen is not None:
            is_ptr = declarator.is_indirect()
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

        if node:
            if declarator.init is not None:
                node._has_default_arg = True
            elif node._has_found_default is True:
                raise RuntimeError("Expected default value for %s" % argname)

        # Check template attribute
        temp = arg.template_arguments
        if arg_typemap and arg_typemap.ntemplate_args > 0:
            if not temp:
                cursor.generate(
                    "Type '{}' requires template argument: {}".format(
                        arg_typemap.name, gen_decl(arg)
                    )
                )
            elif len(temp) != arg_typemap.ntemplate_args:
                cursor.generate(
                    "Type '{}' may only have {} template arguments: {}".format(
                        arg_typemap.name, arg_typemap.ntemplate_args, gen_decl(arg)
                    )
                )
            else:
                for targ in temp:
                    arg_typemap = targ.typemap
                    if arg_typemap is None:
                        # XXX - Not sure this can happen with current parser
                        raise RuntimeError(
                            "check_arg_attr: No such type %s for template: %s"
                            % (temp, gen_decl(arg))
                        )
        elif temp:
            cursor.generate(
                "Type '{}' may not supply template argument: {}".format(
                    arg_typemap.name, gen_decl(arg))
            )

        # Flag node if any argument is assumed-rank.
        dim = attrs.get("dimension", None)  # assumed-rank
        if dim == "..":   # assumed-rank
            node._gen_fortran_generic = True


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
              append_function_index
              has_default_args
              template_function
              define_fortran_generic_functions
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
            found_ctor = found_ctor or declarator.is_ctor
            found_dtor = found_dtor or declarator.is_dtor

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
            nptr = declarator.is_pointer()
            if not options.F_struct_getter_setter:
                return
            elif sgroup == "struct":
                if nptr != 1 and nptr != 2:
                    # Array of pointers
                    return
            elif nptr != 1:
                # Do not write getter for scalar fields since they can be accessed directly
                # Skip scalar and char**.
                return
            elif sgroup in ["char", "string"]:
                # No strings for now.
                return
            # Explicity add the 'this' argument. Added automatically for classes.
            typename = cls.typemap.name
            this_get = "{} *{}".format(typename, cls.fmtdict.CXX_this)
            this_set = this_get + ","   # defaults to intent(inout)
            this_get += "+intent(in)"
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

        ##########
        # getter
        argdecl = declstr.gen_arg_as_language(ast, lang=lang, name=funcname_get)
        decl = "{}({})".format(argdecl, this_get)

        attrs = declarator.attrs
        fattrs = dict(
            intent="getter",
        )
        if "dimension" in attrs:
            fattrs["dimension"] = attrs["dimension"]
        if "deref" in attrs:
            fattrs["deref"] = attrs["deref"]

        fcn = parent.add_function(decl, format=fmt_func, fattrs=fattrs)
        if is_struct:
            fcn.struct_parent = cls
        fcn.wrap.assign(fortran=True)
        fcn._generated = "getter/setter"
        fcn._generated_path.append("getter/setter")

        ##########
        # setter
        if declarator.attrs.get("readonly", False):
            return
        argdecl = declstr.gen_arg_as_language(ast, lang=lang, name="val")
        decl = "void {}({}{})".format(funcname_set, this_set, argdecl)

        fattrs = dict(
            intent="setter",
        )
        attrs = dict(
            val=dict(
                intent="setter",
            )
        )
        if "dimension" in declarator.attrs:
            attrs["val"]["rank"] = declast.find_rank_of_dimension(declarator.attrs["dimension"])

        fcn = parent.add_function(decl, fattrs=fattrs, attrs=attrs, format=fmt_func)
        fcn.wrap.assign(fortran=True)
        fcn._generated = "getter/setter"
        fcn._generated_path.append("getter/setter")

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

    def template_class(self, cls, clslist):
        """Create a class for each list of template_arguments.
        """
        orig_typemap = cls.typemap
        if orig_typemap.cxx_instantiation is None:
            orig_typemap.cxx_instantiation = {}
        for function in cls.functions:
            self.append_function_index(function)
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

            # Add fmt and options from YAML
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

    def share_class(self, cls, smart):
        """Create a subclass for use with smart pointers like std::shared.

        Parameters
        ----------
          cls   - ast.ClassNode
          smart - is a dictionary with fields from YAML file (for smart pointers)
              type   - std::shared_ptr, std::weak_ptr
              format -
        """
        newcls = cls.clone()
        newcls.smart_pointer = []
        fmt_class = newcls.fmtdict

        name_type = smart["type"]
        ntypemap = cls.symtab.lookup_typemap(name_type)
        if ntypemap is None:
            error.get_cursor().warning(
                "smart_pointer type '{}' is unknown".format(name_type))
            return
        if ntypemap.sgroup != "smart_ptr":
            error.get_cursor().warning(
                "smart_pointer type '{}' is not a smart pointer".format(name_type))
            return

        fmt_class.smart_pointer = ntypemap.smart_pointer
        if "format" in smart:
            fmt_class.update(smart["format"])
        newcls.eval_template("C_name_shared_api")
        newcls.name_api = fmt_class.C_name_shared_api
        newcls.name_instantiation = "{}<{}>".format(name_type, fmt_class.cxx_type)
        newcls.scope_file[-1] = newcls.name_api

        newcls.C_shared_class = True
        # Remove defaulted attributes then reset with current values.
        newcls.delete_format_templates()
        newcls.default_format()

        newcls.typemap = declast.fetch_smart_ptr_typemap(
            newcls.name_instantiation,
            ntypemap,
            cls.typemap,
            self.newlibrary.symtab)
        typemap.fill_class_typemap(newcls)
        newcls.typemap.base = "smartptr"
        newcls.typemap.sgroup = "smartptr"

        newcls.baseclass = [ ( 'public', "DDDD", cls.ast) ]
        return newcls

    def add_share_smart_methods(self, cls):
        """A methods to a std::shared_ptr class.

        long use_count() const noexcept;
	T* get() const noexcept;
        """
        fmt_class = cls.fmtdict
        if fmt_class.F_name_shared_use_count:
            decl = "long use_count(void)"
            fmt_func = dict(
                F_name_api = fmt_class.F_name_shared_use_count,
            )
            fcn = cls.add_function(decl, format=fmt_func)
            fcn.C_shared_method = True
        
    def add_smart_ptr_methods(self, smart_ptrs):
        """
        Subclasses for smart pointers have been created.
        This allows one smart pointer to reference another.

        Parameters
        ----------
        smart_ptrs - index of smart pointer subclasses, indexed by type name.
        """
        assign_operators = self.newlibrary.assign_operators

        clsmap = {}
        for cls in smart_ptrs.values():
            clsmap[cls.typemap.smart_pointer] = cls
            
        for cls in smart_ptrs.values():
            ntypemap = cls.typemap
            if ntypemap.smart_pointer == "weak":
                cls.functions = []
                if 'shared' in clsmap:
                    # assign a shared_ptr to a weak_ptr
                    assign_operators.append(AssignOperator(cls, clsmap['shared']))

            self.add_share_smart_methods(cls)
        
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
                self.template_class(cls, clslist)
            else:
                clslist.append(cls)
                self.process_class(cls, cls)
                smart_ptrs = {}
                # cls.smart_pointer is a dict from YAML file.
                for smart in cls.smart_pointer:
                    shared = self.share_class(cls, smart)
                    clslist.append(shared)
                    smart_ptrs[smart["type"]] = shared
                self.add_smart_ptr_methods(smart_ptrs)
                for shared in smart_ptrs.values():
                    self.process_class(shared, shared)
                
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
                        arg.declarator.typemap = typedef[1]
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
        struct_members = {}
        for var in cls.variables:
            a = copy.deepcopy(var.ast)
            if a.declarator.is_pointer():
                a.declarator.attrs["intent"] = "in"
            struct_members[a.declarator.user_name] = var
            ast.declarator.params.append(a)
        node = cls.add_function(name, ast)
        node.declgen = gen_decl(node.ast)
        node.struct_members = struct_members
        node.wrap.assign(python=True)
        node._generated = "struct_as_class_ctor"
        node._generated_path.append("struct_as_class_ctor")

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
        cls.create_node_map()
        for var in cls.variables:
            self.add_var_getter_setter(parent, cls, var)
        cls.functions = self.define_function_suffix(cls.functions)

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
            if function.ast.declarator.is_ctor:
                if not function.wrap.fortran:
                    continue
                # Always create generic interface for class derived type.
                fmt = function.fmtdict
                name = fmt.F_derived_name
                overloaded_functions.setdefault(name, []).append(
                    function)
                function.options.F_create_generic = True
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

        ordered3 = self.define_fortran_generic_functions(ordered_functions)

        self.gen_functions_decl(ordered3)

        return ordered3

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
            new._generated_path.append("cxx_template")

            fmt = new.fmtdict
            if targs.fmtdict:
                fmt.update(targs.fmtdict)
                new.user_fmt = targs.fmtdict

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
        new._generated_path.append("cxx_template")

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
                dim = attrs.get("dimension", None)
                if dim == "..":   # assumed-rank
                    # Replace dimension(..) with rank(n).
                    del attrs["dimension"]
                    attrs["rank"] = rank
            generic = ast.FortranGeneric(
                "", function_suffix="_{}d".format(rank),
                decls=newdecls)
            node.fortran_generic.append(generic)

        # Remove assumed-rank from C function.
        for decl in params:
            attrs = decl.declarator.attrs
            dim = attrs.get("dimension", None)
            if dim == "..":   # assumed-rank
                del attrs["dimension"]
        node.declgen = gen_decl(node.ast)
        
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
        any_need_wrapper = False
        for generic in node.fortran_generic:
            new = node.clone()
            ordered_functions.append(new)
            self.append_function_index(new)
            new._generated = "fortran_generic"
            new._generated_path.append("fortran_generic")
            new.wrap.assign(fortran=True)
            fmt = new.fmtdict
            # XXX append to existing suffix
            if generic.fmtdict:
                fmt.update(generic.fmtdict)
            fmt.function_suffix = fmt.function_suffix + generic.function_suffix
            new.fortran_generic = {}
            new.wrap.assign(fortran=True)
            if len(new.ast.declarator.params) != len(generic.decls):
                raise RuntimeError("internal: generic_function: length mismatch: "
                                   + node.name)
            new.ast.declarator.params = generic.decls

            # Try to call original C function if possible.
            # All arguments are native scalar.
            need_wrapper = False
            if new.return_this:
                pass
            elif new.ast.declarator.is_indirect():
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
                new._PTR_C_CXX_index = node._function_index
                any_need_wrapper = True
            else:
                new.C_fortran_generic = True
                new._PTR_F_C_index = node._function_index
        
        # Do not process templated node, instead process
        # generated functions above.
        #        node.wrap.c = False
        node.wrap.fortran = False

        # At least one of the fortran_generic functions will call
        # The C library function directly.
        if not any_need_wrapper:
            node.wrap.c = True

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

        It is also necessary to trim fortran_generic.
        For example, func() will not have any generic variations
        since it has no arguments.

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
            new._generated_path.append("has_default_arg")
            del new.ast.declarator.params[i:]  # remove trailing arguments
            new._has_default_arg = False
            if node.fortran_generic:
                new.fortran_generic = ast.trim_fortran_generic_decls(node.fortran_generic, i)
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
            node.declgen = gen_decl(node.ast)

def generate_functions(library, config):
    VerifyAttrs(library, config).verify_attrs()
    GenFunctions(library, config).gen_library()
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
            if fcn.ast.declarator.is_ctor:
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
        expr = decl.declarator.attrs.get("implied")
        if expr is not None:
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

def gen_decl(ast):
    """Create str for Declaration and Declarator.
    """
    s = str(ast)
    s2 = str(ast.declarator)
    if ast.is_ctor or ast.is_dtor:
        # Class()  -- no blank afer declaration
        return s + s2
    else:
        # int (*)(void)  -- add blank after declaration
        return s + " " + s2

######################################################################

class AssignOperator(object):
    """
    Used to create an assignment operator overload between two classes.
    """
    def __init__(self, lhs, rhs):
        self.lhs = lhs
        self.rhs = rhs
        self.name = "%s = %s" % (lhs.typemap.name, rhs.typemap.name)
        self.bind = None
        self.cstmt = None
        self.fstmt = None
