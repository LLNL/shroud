# Copyright (c) 2017-2019, Lawrence Livermore National Security, LLC and
# other Shroud Project Developers.
# See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

"""
Abstract Syntax Tree nodes for Library, Class, and Function nodes.
"""
from __future__ import print_function
from __future__ import absolute_import

import copy

from . import util
from . import declast
from . import todict
from . import typemap


class AstNode(object):
    is_class = False

    def option_to_fmt(self):
        """Set fmt based on options dictionary.
        """
        for name in [
            "C_prefix",
            "F_C_prefix",
            "C_result",
            "F_result",
            "F_derived_member",
            "PY_result",
            "LUA_result",
            "C_this",
            "CXX_this",
            "F_this",
            "C_string_result_as_arg",
            "F_string_result_as_arg",
            "C_header_filename_suffix",
            "C_impl_filename_suffix",
            "F_filename_suffix",
            "PY_prefix",
            "PY_header_filename_suffix",
            "PY_impl_filename_suffix",
            "LUA_prefix",
            "LUA_header_filename_suffix",
            "LUA_impl_filename_suffix",
        ]:
            if self.options.inlocal(name):
                raise DeprecationWarning(
                    "Setting option {} for {}, change to format group".format(
                        name, self.__class__.__name__
                    )
                )

    def eval_template(self, name, tname="", fmt=None):
        """If a format has not been explicitly set, set from template."""
        if fmt is None:
            fmt = self.fmtdict
        if not fmt.inlocal(name):
            tname = name + tname + "_template"
            setattr(fmt, name, util.wformat(self.options[tname], fmt))

    def get_LibraryNode(self):
        """Return top of AST tree."""
        return self.parent.get_LibraryNode()


######################################################################


class NamespaceMixin(object):
    def add_class(self, name, template_parameters=None, **kwargs):
        """Add a class.

        template_parameters - list names of template parameters.
             ex. template<typename T>  -> ['T']
        """
        node = ClassNode(
            name, self, template_parameters=template_parameters, **kwargs
        )
        self.classes.append(node)
        self.symbols[name] = node
        return node

    def add_declaration(self, decl, **kwargs):
        """parse decl and add corresponding node.
        decl - declaration

        kwargs -
           cxx_template -
        """
        # parse declaration to find out what it is.
        fullast = declast.check_decl(decl, namespace=self)
        template_parameters = []
        if isinstance(fullast, declast.Template):
            # Create list of template parameter names
            # template<typename T> class vector -> ['T']
            for tparam in fullast.parameters:
                template_parameters.append(tparam.name)
            ast = fullast.decl
        else:
            ast = fullast

        if isinstance(ast, declast.Declaration):
            if "typedef" in ast.storage:
                node = self.create_typedef_typemap(ast, **kwargs)
            elif ast.params is None:
                node = self.add_variable(decl, ast=ast, **kwargs)
            else:
                node = self.add_function(decl, ast=fullast, **kwargs)
        elif isinstance(ast, declast.CXXClass):
            # A Class may already be forwared defined.
            # If so, just return it.
            node = self.symbols.get(ast.name, None)
            if not node:
                node = self.add_class(
                    ast.name, template_parameters=template_parameters, **kwargs
                )
        elif isinstance(ast, declast.Namespace):
            node = self.add_namespace(ast.name, **kwargs)
        elif isinstance(ast, declast.Enum):
            node = self.add_enum(decl, ast=ast, **kwargs)
        elif isinstance(ast, declast.Struct):
            node = self.add_struct(decl, ast=ast, **kwargs)
        else:
            raise RuntimeError(
                "add_declaration: unknown ast type {} after parsing '{}'".format(
                    type(ast), decl
                )
            )
        return node

    def create_typedef_typemap(self, ast, **kwargs):
        """Create a TypedefNode from a Declarator.
        """
        if ast.declarator.pointer:
            raise NotImplementedError("Pointers not supported in typedef")
        if ast.declarator.func:
            raise NotImplementedError(
                "Function pointers not supported in typedef"
            )

        key = ast.declarator.name
        orig = ast.typemap
        ntypemap = orig.clone_as(self.scope + key)
        ntypemap.typedef = orig.name
        ntypemap.cxx_type = ntypemap.name
        ntypemap.compute_flat_name()
        if "fields" in kwargs:
            ntypemap.update(kwargs["fields"])
        typemap.register_type(ntypemap.name, ntypemap)

        node = self.add_typedef(key, ntypemap)
        return node

    def add_enum(self, decl, ast=None, **kwargs):
        """Add an enumeration.
        """
        node = EnumNode(decl, parent=self, ast=ast, **kwargs)
        self.enums.append(node)
        self.symbols[node.name] = node
        return node

    def add_function(self, decl, ast=None, **kwargs):
        """Add a function.

        decl - C/C++ declaration of function
        ast  - parsed declaration. None if not yet parsed.
        """
        fcnnode = FunctionNode(decl, parent=self, ast=ast, **kwargs)
        self.functions.append(fcnnode)
        return fcnnode

    def add_namespace(self, name, **kwargs):
        """Add an namespace
        """
        node = NamespaceNode(name, parent=self, **kwargs)
        self.symbols[name] = node
        return node

    def add_namespaces(self, names):
        """Create nested namespaces from list of names.
        """
        ns = self
        for name in names:
            if name in ns.symbols:
                ns = ns.symbols[name]
            else:
                ns = ns.add_namespace(name)
        return ns

    def add_struct(self, decl, ast=None, **kwargs):
        """Add a struct.
        A struct is exactly like a class to the C++ compiler.
        From the YAML, a struct is a single ast and a class is broken into parts.
        """
        name = ast.name
        node = ClassNode(name, self, as_struct=True, **kwargs)
        for member in ast.members:
            node.add_variable(str(member), member)
        self.classes.append(node)
        self.symbols[node.name] = node
        return node

    def add_typedef(self, name, ntypemap=None):
        """Add a TypedefNode to the symbol table.
        """
        node = TypedefNode(name, parent=self, ntypemap=ntypemap)
        self.symbols[name] = node
        return node

    def add_variable(self, decl, ast=None, **kwargs):
        """Add a variable or class member.

        decl - C/C++ declaration of function
        ast  - parsed declaration. None if not yet parsed.
        """
        node = VariableNode(decl, parent=self, ast=ast, **kwargs)
        self.variables.append(node)
        return node


######################################################################


class LibraryNode(AstNode, NamespaceMixin):
    """There is one library per wrapping.
    It represents the global namespace.
    """
    def __init__(
        self,
        cxx_header="",
        format=None,
        language="c++",
        library="default_library",
        options=None,
        **kwargs
    ):
        """Create LibraryNode.

        cxx_header = blank delimited list of headers for C++ or C library.

        fields = value
        options:
        classes:
        functions:

        """
        # From arguments
        self.cxx_header = cxx_header
        self.language = language.lower()
        if self.language not in ["c", "c++"]:
            raise RuntimeError("language must be 'c' or 'c++'")
        if self.language == "c++":
            # Use a form which can be used as a variable name
            self.language = "cxx"
        self.library = library

        self.classes = []
        self.enums = []
        self.functions = []
        self.variables = []
        # Each is given a _function_index when created.
        self.function_index = []
        self.options = self.default_options()
        if options:
            self.options.update(options, replace=True)

        self.F_module_dependencies = []  # unused

        self.copyright = kwargs.get("copyright", [])
        self.patterns = kwargs.get("patterns", [])

        self.default_format(format, kwargs)

        # namespace
        self.scope = ""
        self.symbols = {}
        self.using = []
        declast.global_namespace = self
        self.create_std_names()

    def get_LibraryNode(self):
        """Return top of AST tree."""
        return self

    # # # # # namespace behavior

    def create_std_names(self):
        """Add standard types to the Library."""
        self.add_typedef("size_t")
        self.add_typedef("int8_t")
        self.add_typedef("int16_t")
        self.add_typedef("int32_t")
        self.add_typedef("int64_t")
        self.add_typedef("uint8_t")
        self.add_typedef("uint16_t")
        self.add_typedef("uint32_t")
        self.add_typedef("uint64_t")
        self.add_typedef("MPI_Comm")
        create_std_namespace(self)  # add 'std::' to library
        self.using_directive("std")

    def qualified_lookup(self, name):
        """Look for symbols within class.
        """
        return self.symbols.get(name, None)

    def unqualified_lookup(self, name):
        """Look for symbols within library (global namespace). """
        if name in self.symbols:
            return self.symbols[name]
        for ns in self.using:
            item = ns.qualified_lookup(name)
            if item is not None:
                return item
        return None

    def using_directive(self, name):
        """Implement 'using namespace <name>'
        """
        ns = self.unqualified_lookup(name)
        if ns is None:
            raise RuntimeError("{} not found in namespace".format(name))
        if ns not in self.using:
            self.using.append(ns)

    def add_shadow_typemap(self, ntypemap):
        """Add a shadow typemap into the symbol table.
        ntypemap is created by create_class_typemap_from_fields
        using data from the YAML file.
        Adding to the symbol table allows it to be parsed.

        cxx_name is always fully qualified (namespace1::namespace2::class)
        """
        names = ntypemap.name.split("::")
        cxx_name = names.pop()
        ns = self.add_namespaces(names)

        node = ClassNode(cxx_name, ns, ntypemap=ntypemap)
        # node is not added to self.classes
        ns.symbols[cxx_name] = node

    #####

    def default_options(self):
        """default options."""
        def_options = util.Scope(
            parent=None,
            debug=False,  # print additional debug info
            debug_index=False,  # print function indexes. debug must also be True.
            # They change when a function is inserted.
            C_line_length=72,
            F_line_length=72,
            F_module_per_class=True,
            F_string_len_trim=True,
            F_force_wrapper=False,
            F_return_fortran_pointer=True,
            F_standard=2003,
            F_auto_reference_count=False,
            F_create_bufferify_function=True,
            wrap_c=True,
            wrap_fortran=True,
            wrap_python=False,
            wrap_lua=False,
            doxygen=True,  # create doxygen comments
            literalinclude=False, # Create sphinx literalinclude markers
            return_scalar_pointer="pointer",
            show_splicer_comments=True,
            # blank for functions, set in classes.
            class_prefix_template="{class_lower}_",
            YAML_type_filename_template="{library_lower}_types.yaml",
            C_header_filename_library_template="wrap{library}.{C_header_filename_suffix}",
            C_impl_filename_library_template="wrap{library}.{C_impl_filename_suffix}",
            C_header_filename_class_template="wrap{cxx_class}.{C_header_filename_suffix}",
            C_impl_filename_class_template="wrap{cxx_class}.{C_impl_filename_suffix}",
            C_header_utility_template="types{library}.{C_header_filename_suffix}",
            C_enum_template="{C_prefix}{flat_name}",
            C_enum_member_template="{C_prefix}{C_scope_name}{enum_member_name}",
            C_name_template=(
                "{C_prefix}{class_prefix}{underscore_name}{function_suffix}{template_suffix}"
            ),
            C_memory_dtor_function_template=(
                "{C_prefix}SHROUD_memory_destructor"
            ),
            C_var_capsule_template="C{c_var}",  # capsule argument
            C_var_context_template="D{c_var}",  # context argument
            C_var_len_template="N{c_var}",  # argument for result of len(arg)
            C_var_trim_template="L{c_var}",  # argument for result of len_trim(arg)
            C_var_size_template="S{c_var}",  # argument for result of size(arg)
            # Fortran's names for C functions
            F_C_name_template=(
                "{F_C_prefix}{class_prefix}{underscore_name}{function_suffix}{template_suffix}"
            ),
            F_enum_member_template=(
                "{F_scope_name}{enum_member_lower}"
            ),
            F_name_impl_template=(
                "{class_prefix}{underscore_name}{function_suffix}{template_suffix}"
            ),
            F_name_function_template="{underscore_name}{function_suffix}{template_suffix}",
            F_name_generic_template="{underscore_name}",
            F_module_name_library_template="{library_lower}_mod",
            F_impl_filename_library_template="wrapf{library_lower}.{F_filename_suffix}",
            F_capsule_data_type_class_template="SHROUD_{class_lower}_capsule",
            F_module_name_class_template="{class_lower}_mod",
            F_impl_filename_class_template="wrapf{cxx_class}.{F_filename_suffix}",
            F_abstract_interface_subprogram_template="{underscore_name}_{argname}",
            F_abstract_interface_argument_template="arg{index}",

            LUA_module_name_template="{library_lower}",
            LUA_module_filename_template=(
                "lua{library}module.{LUA_impl_filename_suffix}"
            ),
            LUA_header_filename_template=(
                "lua{library}module.{LUA_header_filename_suffix}"
            ),
            LUA_userdata_type_template="{LUA_prefix}{cxx_class}_Type",
            LUA_userdata_member_template="self",
            LUA_module_reg_template="{LUA_prefix}{library}_Reg",
            LUA_class_reg_template="{LUA_prefix}{cxx_class}_Reg",
            LUA_metadata_template="{cxx_class}.metatable",
            LUA_ctor_name_template="{cxx_class}",
            LUA_name_template="{function_name}",
            LUA_name_impl_template="{LUA_prefix}{class_prefix}{underscore_name}",

            PY_module_filename_template=(
                "py{library}module.{PY_impl_filename_suffix}"
            ),
            PY_header_filename_template=(
                "py{library}module.{PY_header_filename_suffix}"
            ),
            PY_utility_filename_template=(
                "py{library}util.{PY_impl_filename_suffix}"
            ),
            PY_PyTypeObject_template="{PY_prefix}{cxx_class}_Type",
            PY_PyObject_template="{PY_prefix}{cxx_class}",
            PY_type_filename_template=(
                "py{cxx_class}type.{PY_impl_filename_suffix}"
            ),
            PY_name_impl_template=(
                "{PY_prefix}{class_prefix}{function_name}{function_suffix}{template_suffix}"
            ),
            # names for type methods (tp_init)
            PY_type_impl_template=(
                "{PY_prefix}{cxx_class}_{PY_type_method}{function_suffix}{template_suffix}"
            ),
            PY_member_getter_template=(
                "{PY_prefix}{cxx_class}_{variable_name}_getter"
            ),
            PY_member_setter_template=(
                "{PY_prefix}{cxx_class}_{variable_name}_setter"
            ),
            PY_struct_array_descr_create_template=(
                "{PY_prefix}{cxx_class}_create_array_descr"
            ),
            PY_struct_array_descr_variable_template=(
                "{PY_prefix}{cxx_class}_array_descr"
            ),
            PY_struct_array_descr_name_template=("{cxx_class}_dtype"),
            PY_numpy_array_capsule_name_template=("{PY_prefix}array_dtor"),
            PY_dtor_context_array_template=(
                # array of PY_dtor_context_typedef
                "{PY_prefix}SHROUD_capsule_context"
            ),
            PY_dtor_context_typedef_template=(
                "{PY_prefix}SHROUD_dtor_context"
            ),
            PY_capsule_destructor_function_template=(
                "{PY_prefix}SHROUD_capsule_destructor"
            ),
            PY_release_memory_function_template=(
                "{PY_prefix}SHROUD_release_memory"
            ),
            PY_fetch_context_function_template=(
                "{PY_prefix}SHROUD_fetch_context"
            ),
            PY_array_arg="numpy",   # or "list"
            PY_struct_arg="numpy",   # or "list", "class"
        )

        return def_options

    def default_format(self, fmtdict, kwargs):
        """Set format dictionary.

        Values based off of library variables and
        format templates in options.
        """

        C_prefix = self.library.upper()[:3] + "_"  # function prefix
        fmt_library = util.Scope(
            parent=None,
            C_bufferify_suffix="_bufferify",
            C_prefix=C_prefix,
            C_result="rv",  # return value
            C_argument="SH_",
            c_temp="SHT_",
            C_local="SHC_",
            C_this="self",
            C_custom_return_type="",  # assume no value
            CXX_this="SH_this",
            CXX_local="SHCXX_",
            cxx_class="",  # Assume no class
            class_scope="",
            F_C_prefix="c_",
            F_derived_member="cxxmem",
            F_name_assign="assign",
            F_name_associated="associated",
            F_name_instance_get="get_instance",
            F_name_instance_set="set_instance",
            F_name_final="final",
            F_result="SHT_rv",
            F_result_ptr="SHT_prv",
            F_result_capsule="SHT_crv",
            F_pointer="SHT_ptr",
            F_this="obj",
            C_string_result_as_arg="SHF_rv",
            F_string_result_as_arg="",
            C_capsule_data_type=C_prefix + "SHROUD_capsule_data",
            F_capsule_data_type="SHROUD_capsule_data",
            F_capsule_type="SHROUD_capsule",
            F_capsule_final_function="SHROUD_capsule_final",
            C_array_type=C_prefix + "SHROUD_array",
            F_array_type="SHROUD_array",

            LUA_result="rv",
            LUA_prefix="l_",
            LUA_state_var="L",
            LUA_this_call="",

            PY_prefix="PY_",
            PY_module_name=self.library.lower(),
            PY_result="SHTPy_rv",  # Create PyObject for result
            PY_this_call="",
            PY_type_obj="obj",  # name of cpp class pointer in PyObject
            PY_type_dtor="idtor",  # name of destructor capsule infomation

            library=self.library,
            library_lower=self.library.lower(),
            library_upper=self.library.upper(),
            # set default values for fields which may be unset.
            class_prefix="",  # expand to blanks for library
            # c_ptr='',
            # c_const='',
            CXX_this_call="",
            CXX_template="",
            C_pre_call="",
            C_post_call="",
            function_suffix="",  # assume no suffix
            template_suffix="",  # assume no suffix
            namespace_scope="",
        )

        fmt_library.F_filename_suffix = "f"

        if self.language == "c":
            fmt_library.C_header_filename_suffix = "h"
            fmt_library.C_impl_filename_suffix = "c"

            fmt_library.LUA_header_filename_suffix = "h"
            fmt_library.LUA_impl_filename_suffix = "c"

            fmt_library.stdlib = ""
            fmt_library.void_proto = "void"

            fmt_library.cast_const = "("
            fmt_library.cast_reinterpret = "("
            fmt_library.cast_static = "("
            fmt_library.cast1 = ") "
            fmt_library.cast2 = ""
        else:
            fmt_library.C_header_filename_suffix = "h"
            fmt_library.C_impl_filename_suffix = "cpp"

            fmt_library.LUA_header_filename_suffix = "hpp"
            fmt_library.LUA_impl_filename_suffix = "cpp"

            fmt_library.stdlib = "std::"
            fmt_library.void_proto = ""

            fmt_library.cast_const = "const_cast<"
            fmt_library.cast_reinterpret = "reinterpret_cast<"
            fmt_library.cast_static = "static_cast<"
            fmt_library.cast1 = ">\t("
            fmt_library.cast2 = ")"

        for name in [
            "C_header_filename",
            "C_impl_filename",
            "F_module_name",
            "F_impl_filename",
            "LUA_module_name",
            "LUA_module_reg",
            "LUA_module_filename",
            "LUA_header_filename",
            "PY_module_filename",
            "PY_header_filename",
            "PY_helper_filename",
            "YAML_type_filename",
        ]:
            if name in kwargs:
                raise DeprecationWarning(
                    "Setting field {} in library, change to format group".format(
                        name
                    )
                )

        self.option_to_fmt()

        if fmtdict:
            fmt_library.update(fmtdict, replace=True)

        self.fmtdict = fmt_library

        # default some format strings based on other format strings
        self.eval_template("C_header_filename", "_library")
        self.eval_template("C_impl_filename", "_library")
        self.eval_template("C_header_utility")

        self.eval_template("C_memory_dtor_function")

        # All class/methods and functions may go into this file or
        # just functions.
        self.eval_template("F_module_name", "_library")
        self.eval_template("F_impl_filename", "_library")

        self.eval_template("PY_numpy_array_capsule_name")
        self.eval_template("PY_dtor_context_array")
        self.eval_template("PY_dtor_context_typedef")
        self.eval_template("PY_capsule_destructor_function")
        self.eval_template("PY_release_memory_function")
        self.eval_template("PY_fetch_context_function")


######################################################################


class BlockNode(AstNode, NamespaceMixin):
    """Create a Node to simulate a curly block.
    A block can contain options, format, and declarations.
    The declarations within a BlockNode inherit options of the block.
    This makes it easier to change options for a group of functions.
    Declarations are added to parent.

    Blocks can be added to a LibraryNode, NamespaceNode or ClassNode.
    """

    def __init__(self, parent, format=None, options=None, **kwargs):
        # From arguments
        self.parent = parent

        self.enums = parent.enums
        self.functions = parent.functions
        self.classes = parent.classes
        self.variables = parent.variables

        self.options = util.Scope(parent=parent.options)
        if options:
            self.options.update(options, replace=True)

        self.fmtdict = util.Scope(parent=parent.fmtdict)
        if format:
            self.fmtdict.update(format, replace=True)

    def unqualified_lookup(self, name):
        """Look for symbols within parent. """
        return self.parent.unqualified_lookup(name)


######################################################################


class NamespaceNode(AstNode, NamespaceMixin):
    def __init__(self, name, parent, format=None, options=None, **kwargs):
        """Create NamespaceNode.

        parent may be LibraryNode or NamespaceNode.
        """
        # From arguments
        self.name = name
        self.parent = parent
        self.linenumber = kwargs.get("__line__", "?")

        # Namespaces do not own enums, functions or classes directly.
        # Find their owner up the parent chain.
        owner = parent
        while owner:
            if isinstance(owner, NamespaceNode):
                # skip over nested namespaces
                owner = owner.parent
            self.enums = owner.enums
            self.functions = owner.functions
            self.classes = owner.classes
            self.variables = owner.variables
            break

        self.options = util.Scope(parent=parent.options)
        if options:
            self.options.update(options, replace=True)

        self.default_format(parent, format)

        # add to symbol table
        self.scope = self.parent.scope + self.name + "::"
        self.symbols = {}
        self.using = []

    # # # # # namespace behavior

    def qualified_lookup(self, name):
        """Look for symbols within class.
        -- Only enums
        """
        return self.symbols.get(name, None)

    def unqualified_lookup(self, name):
        """Look for symbols within library (global namespace)."""
        if name in self.symbols:
            return self.symbols[name]
        for ns in self.using:
            item = ns.unqualified_lookup(name)
            if item is not None:
                return item
        return self.parent.unqualified_lookup(name)

    def using_directive(self, name):
        """Implement 'using namespace <name>'
        """
        ns = self.unqualified_lookup(name)
        if ns is None:
            raise RuntimeError("{} not found in namespace".format(name))
        if ns not in self.using:
            self.using.append(ns)

    #####

    def default_format(self, parent, format):
        """Set format dictionary."""

        self.fmtdict = util.Scope(parent=parent.fmtdict)

        fmt_ns = self.fmtdict
        fmt_ns.namespace_scope = (
            parent.fmtdict.namespace_scope + self.name + "::"
        )
        fmt_ns.CXX_this_call = fmt_ns.namespace_scope
        fmt_ns.LUA_this_call = fmt_ns.namespace_scope
        fmt_ns.PY_this_call = fmt_ns.namespace_scope

        if format:
            fmt_ns.update(format, replace=True)


######################################################################


class ClassNode(AstNode, NamespaceMixin):
    """A C++ class or struct.

    """

    is_class = True

    def __init__(
        self,
        name,
        parent,
        cxx_header="",
        format=None,
        options=None,
        as_struct=False,
        template_parameters=None,
        ntypemap=None,
        **kwargs
    ):
        """Create ClassNode.
        Used with class or struct if as_struct==True.

        template_parameters - list names of template parameters.
             ex. template<typename T>  -> ['T']
        Added to symbol table.

        cxx_template - list of TemplateArgument instances
        """
        # From arguments
        self.name = name
        self.parent = parent
        self.cxx_header = cxx_header
        self.linenumber = kwargs.get("__line__", "?")

        self.enums = []
        self.functions = []
        self.variables = []
        self.as_struct = (
            as_struct
        )  # if True, treat as struct, else as shadow class

        self.python = kwargs.get("python", {})
        self.cpp_if = kwargs.get("cpp_if", None)

        self.options = util.Scope(parent=parent.options)
        if options:
            self.options.update(options, replace=True)

        self.default_format(parent, format, kwargs)

        # Add to namespace.
        self.scope = self.parent.scope + self.name + "::"
        self.symbols = {}

        fields = kwargs.get("fields", None)
        if fields is not None:
            if not isinstance(fields, dict):
                raise TypeError("fields must be a dictionary")
        if ntypemap is not None:
            # From YAML typemap
            self.typemap = ntypemap
        elif as_struct:
            self.typemap = typemap.create_struct_typemap(self, fields)
        else:
            self.typemap = typemap.create_class_typemap(self, fields)
        if format and 'template_suffix' in format:
            # Do not use scope from self.fmtdict, instead only copy value
            # when in the format dictionary is passed in.
            self.typemap.template_suffix = format['template_suffix']

        # Add template parameters.
        if template_parameters is None:
            self.template_parameters = []
        else:
            self.template_parameters = template_parameters
            for param_name in template_parameters:
                self.create_template_parameter_typemap(param_name)

        # Parse the instantiations.
        # cxx_template = [ TemplateArgument('<int>'),
        #                  TemplateArgument('<double>') ]
        cxx_template = kwargs.get("cxx_template", [])
        self.template_arguments = cxx_template
        for args in cxx_template:
            args.parse_instantiation(namespace=self)

    # # # # # namespace behavior

    def create_template_parameter_typemap(self, name):
        """Create a typemap for a template parameter.
        Use base='template'.

        The real type will be used during template instantiation.
        """
        fullname = self.scope + name
        ntypemap = typemap.Typemap(
            fullname,
            base="template",
            c_type="c_T",
            cxx_type="cxx_T",
            f_type="f_T",
        )
        typemap.register_type(ntypemap.name, ntypemap)

        self.add_typedef(name, ntypemap=ntypemap)

    def qualified_lookup(self, name):
        """Look for symbols within class.
        -- Only enums
        """
        return self.symbols.get(name, None)

    def unqualified_lookup(self, name):
        """Look for name in class or its parents.
        Nested classes, namespaces, or library."""
        if name in self.symbols:
            return self.symbols[name]
        return self.parent.unqualified_lookup(name)

    #####

    def default_format(self, parent, format, kwargs):
        """Set format dictionary."""

        for name in [
            "C_header_filename",
            "C_impl_filename",
            "F_derived_name",
            "F_impl_filename",
            "F_module_name",
            "LUA_userdata_type",
            "LUA_userdata_member",
            "LUA_class_reg",
            "LUA_metadata",
            "LUA_ctor_name",
            "PY_PyTypeObject",
            "PY_PyObject",
            "PY_type_filename",
            "class_prefix",
        ]:
            if name in kwargs:
                raise DeprecationWarning(
                    "Setting field {} in class {}, change to format group".format(
                        name, self.name
                    )
                )

        self.fmtdict = util.Scope(
            parent=parent.fmtdict,
            cxx_type=self.name,
            cxx_class=self.name,
            class_lower=self.name.lower(),
            class_upper=self.name.upper(),
            class_scope=self.name + "::",
            F_derived_name=self.name.lower(),
        )

        fmt_class = self.fmtdict
        if format:
            fmt_class.update(format, replace=True)
        self.expand_format_templates()

    def expand_format_templates(self):
        """Expand format templates for a class.
        Called after other format fields are set.
        eval_template will only set a value if it has not already been set.
        Call delete_format_template to remove previous values to
        force them to be recomputed during class template instantiation.
        """
        self.eval_template("class_prefix")

        # Only one file per class for C.
        self.eval_template("C_header_filename", "_class")
        self.eval_template("C_impl_filename", "_class")

        self.eval_template("F_capsule_data_type", "_class")
        if self.options.F_module_per_class:
            self.eval_template("F_module_name", "_class")
            self.eval_template("F_impl_filename", "_class")

        # As PyArray_Descr
        if self.as_struct:
            self.eval_template("PY_struct_array_descr_create")
            self.eval_template("PY_struct_array_descr_variable")
            self.eval_template("PY_struct_array_descr_name")

    def delete_format_templates(self):
        """Delete some format strings which were defaulted.
        Used when instantiation a class to remove previous class' values.
        """
        self.fmtdict.delattrs(
            [
                "class_prefix",
                "C_header_filename",
                "C_impl_filename",
                "F_module_name",
                "F_impl_filename",
                "PY_struct_array_descr_create",
                "PY_struct_array_descr_variable",
                "PY_struct_array_descr_name",
            ]
        )

    def add_namespace(self, **kwargs):
        """Replace method inherited from NamespaceMixin."""
        raise RuntimeError("Cannot add a namespace to a class")

    def clone(self):
        """Create a copy of a ClassNode to use with C++ template.

        Create a clone of fmtdict and options allowing them
        to be modified.
        Clone all functions and reparent fmtdict and options to
        the new class.
        """
        # Shallow copy everything.
        new = copy.copy(self)

        # Add new format and options Scope.
        new.fmtdict = self.fmtdict.clone()
        new.options = self.options.clone()

        # Clone all functions.
        newfcns = []
        for fcn in self.functions:
            newfcn = fcn.clone()
            newfcn.fmtdict.reparent(new.fmtdict)
            newfcn.options.reparent(new.options)
            newfcns.append(newfcn)
        new.functions = newfcns

        return new


######################################################################


class FunctionNode(AstNode):
    """

    - decl: template<typename T1, typename T2> foo(T1 arg, T2 arg)
      cxx_template:
      - instantiation: <int, long>
      - instantiation: <float, double>
      fattrs:     # function attributes
      attrs:
        arg1:     # argument attributes


    _fmtfunc = Scope()

    _fmtresult = {
       'fmtc': Scope(_fmtfunc)
    }
    _fmtargs = {
      'arg1': {
        'fmtc': Scope(_fmtfunc),
        'fmtf': Scope(_fmtfunc)
        'fmtl': Scope(_fmtfunc)
        'fmtpy': Scope(_fmtfunc)
      }
    }

    statements = {
      'c': {
         'result_buf':
       },
       'f': {
       },
    }

    _function_index  - sequence number function,
                       used in lieu of a pointer
    _generated       - who generated this function
    _PTR_F_C_index   - Used by fortran wrapper to find index of
                       C function to call
    _PTR_C_CXX_index - Used by C wrapper to find index of C++ function
                       to call

    Templates
    ---------

    template_parameters - [ 'T1', 'T2' ]
    template_argument - [ TemplateArgument('<int,long>'),
                          TemplateArgument('<float,double>') ]


    """

    def __init__(
        self, decl, parent, format=None, ast=None, options=None, **kwargs
    ):
        """
        ast - None, declast.Declaration, declast.Template
        """
        self.parent = parent
        self.linenumber = kwargs.get("__line__", "?")

        self.options = util.Scope(parent.options)
        if options:
            self.options.update(options, replace=True)

        self.default_format(parent, format, kwargs)

        # working variables
        self._CXX_return_templated = False
        self._PTR_C_CXX_index = None
        self._PTR_F_C_index = None
        self._cxx_overload = None
        self.declgen = None  # generated declaration.
        self._default_funcs = []  # generated default value functions  (unused?)
        self._function_index = None
        self._fmtargs = {}
        self._fmtresult = {}
        self._function_index = None
        self._generated = False
        self._has_default_arg = False
        self._nargs = None
        self._overloaded = False

        # self.function_index = []

        self.default_arg_suffix = kwargs.get("default_arg_suffix", [])
        self.cpp_if = kwargs.get("cpp_if", None)
        self.cxx_template = {}
        self.template_parameters = []
        self.template_arguments = kwargs.get("cxx_template", [])
        self.doxygen = kwargs.get("doxygen", {})
        self.fortran_generic = kwargs.get("fortran_generic", {})
        self.return_this = kwargs.get("return_this", False)

        # Generated by Preprocess
        self.CXX_subprogram = "--none--"
        self.C_subprogram = "--none--"
        self.F_subprogram = "--none--"
        self.CXX_return_type = "--none--"
        self.C_return_type = "--none--"
        self.F_return_type = "--none--"

        # Used with c_statements to find correct intent block
        # possible values are '', '_buf'
        self.generated_suffix = ""

        if not decl:
            raise RuntimeError("FunctionNode missing decl")

        self.decl = decl
        if ast is None:
            ast = declast.check_decl(decl, namespace=parent)
        if isinstance(ast, declast.Template):
            for param in ast.parameters:
                self.template_parameters.append(param.name)

            template_parameters = ast
            ast = ast.decl
            for args in self.template_arguments:
                args.parse_instantiation(namespace=self)

            # XXX - convert to cxx_template format  { T=['int', 'double'] }
            # XXX - only deals with single template argument  [0]?
            argname = template_parameters.parameters[0].name
            lst = []
            for arg in self.template_arguments:
                lst.append(arg.asts[0].typemap.name)
            self.cxx_template[argname] = lst
        elif isinstance(ast, declast.Declaration):
            pass
        else:
            raise RuntimeError("Expected a function declaration")
        if ast.params is None:
            # 'void foo' instead of 'void foo()'
            raise RuntimeError("Missing arguments to function:", ast.gen_decl())
        self.ast = ast

        # Look for any template (include class template) arguments.
        self.have_template_args = False
        if ast.typemap.base == "template":
            self.have_template_args = True
        else:
            for args in ast.params:
                if args.typemap.base == "template":
                    self.have_template_args = True
                    break

        # add any attributes from YAML files to the ast
        if "attrs" in kwargs:
            attrs = kwargs["attrs"]
            for arg in ast.params:
                name = arg.name
                if name in attrs:
                    arg.attrs.update(attrs[name])
        if "fattrs" in kwargs:
            ast.attrs.update(kwargs["fattrs"])
        # XXX - waring about unused fields in attrs

        fmt_func = self.fmtdict
        fmt_func.function_name = ast.name
        fmt_func.underscore_name = util.un_camel(fmt_func.function_name)

    def default_format(self, parent, fmtdict, kwargs):

        # Move fields from kwargs into instance
        for name in [
            "C_code",
            # 'C_error_pattern',
            "C_name",
            "C_post_call",
            "C_post_call_buf",
            "C_return_code",
            "C_return_type",
            "F_C_name",
            "F_code",
            "F_name_function",
            "F_name_generic",
            "F_name_impl",
            "LUA_name",
            "LUA_name_impl",
            # 'PY_error_pattern',
            "PY_name_impl",
            "function_suffix",
        ]:
            if name in kwargs:
                raise DeprecationWarning(
                    "Setting field {} in function, change to format group".format(
                        name
                    )
                )

        # Move fields from kwargs into instance
        for name in ["C_error_pattern", "PY_error_pattern"]:
            setattr(self, name, kwargs.get(name, None))

        self.fmtdict = util.Scope(parent.fmtdict)

        self.option_to_fmt()
        if fmtdict:
            self.fmtdict.update(fmtdict, replace=True)
            if "C_return_type" in fmtdict:
                # wrapc.py will overwrite C_return_type.
                # keep original value for wrapf.py.
                self.fmtdict.C_custom_return_type = fmtdict["C_return_type"]

    def clone(self):
        """Create a copy of a FunctionNode to use with C++ template
        or changing result to argument.
        """
        # Shallow copy everything.
        new = copy.copy(self)

        # new Scope with same inlocal and parent.
        new.fmtdict = self.fmtdict.clone()
        new.options = self.options.clone()

        # Deep copy dictionaries.
        new.ast = copy.deepcopy(self.ast)
        new._fmtargs = copy.deepcopy(self._fmtargs)
        new._fmtresult = copy.deepcopy(self._fmtresult)

        return new

    def unqualified_lookup(self, name):
        """Look for symbols within parent. """
        return self.parent.unqualified_lookup(name)


######################################################################


class EnumNode(AstNode):
    """
        - decl: |
              enum Color {
                RED,
                BLUE,
                WHITE
              }
          options:
             bar: 4
          format:
             baz: 4

    _fmtmembers = {
      'RED': Scope(_fmt_func)

    }
    """

    def __init__(
        self, decl, parent, format=None, ast=None, options=None, **kwargs
    ):

        # From arguments
        self.parent = parent
        self.linenumber = kwargs.get("__line__", "?")

        self.options = util.Scope(parent.options)
        if options:
            self.options.update(options, replace=True)

        #        self.default_format(parent, format, kwargs)
        self.fmtdict = util.Scope(parent=parent.fmtdict)

        if not decl:
            raise RuntimeError("EnumNode missing decl")

        self.decl = decl
        if ast is None:
            ast = declast.check_decl(decl)
        if not isinstance(ast, declast.Enum):
            raise RuntimeError("Declaration is not an enumeration: " + decl)
        self.ast = ast
        self.name = ast.name

        # format for enum
        fmt_enum = self.fmtdict
        fmt_enum.enum_name = ast.name
        fmt_enum.enum_lower = ast.name.lower()
        fmt_enum.enum_upper = ast.name.upper()
        if fmt_enum.cxx_class:
            fmt_enum.namespace_scope = (
                fmt_enum.namespace_scope + fmt_enum.cxx_class + "::"
            )

        # format for each enum member
        fmtmembers = {}
        evalue = 0
        for member in ast.members:
            fmt = util.Scope(parent=fmt_enum)
            fmt.enum_member_name = member.name
            fmt.enum_member_lower = member.name.lower()
            fmt.enum_member_upper = member.name.upper()

            # evaluate value
            if member.value is not None:
                fmt.cxx_value = todict.print_node(member.value)
                evalue = int(todict.print_node(member.value))
            fmt.evalue = evalue
            evalue = evalue + 1

            fmtmembers[member.name] = fmt
        self._fmtmembers = fmtmembers

        # Add to namespace
        self.scope = self.parent.scope + self.name + "::"
        self.typemap = typemap.create_enum_typemap(self)
        # also 'enum class foo' will alter scope

        fmt_enum.flat_name = self.typemap.flat_name
        if fmt_enum.flat_name == self.typemap.name and \
           ast.scope is None:
            # enum Color { RED };
            # {C_prefix}_RED
            fmt_enum.C_scope_name = ""
        else:
            # enum class Color { RED };
            # {C_prefix}_Color_RED
            fmt_enum.C_scope_name = self.typemap.flat_name + "_"

        # XXX - fortran should not include scope levels which relate to 
        # the namespace for the module
        fmt_enum.F_scope_name = fmt_enum.C_scope_name.lower()


######################################################################


class TypedefNode(AstNode):
    """
    Used for namespace resolution

    type name must be in a typemap.
    """

    def __init__(self, name, parent, ntypemap=None):

        # From arguments
        self.name = name
        self.parent = parent

        # Add to namespace
        if ntypemap is None:
            typename = self.parent.scope + self.name
            self.typemap = typemap.lookup_type(typename)
        else:
            self.typemap = ntypemap

    def get_typename(self):
        return self.typemap.name


######################################################################


class VariableNode(AstNode):
    """
        - decl: int var
          options:
             bar: 4
          format:
             baz: 4
    """

    def __init__(
        self, decl, parent, format=None, ast=None, options=None, **kwargs
    ):

        # From arguments
        self.parent = parent
        self.linenumber = kwargs.get("__line__", "?")

        self.options = util.Scope(parent=parent.options)
        if options:
            self.options.update(options, replace=True)

        #        self.default_format(parent, format, kwargs)
        self.fmtdict = util.Scope(parent=parent.fmtdict)

        if not decl:
            raise RuntimeError("VariableNode missing decl")

        self.decl = decl
        if ast is None:
            ast = declast.check_decl(decl)
        if not isinstance(ast, declast.Declaration):
            raise RuntimeError("Declaration is not a structure: " + decl)
        if ast.params is not None:
            # 'void foo()' instead of 'void foo'
            raise RuntimeError("Arguments given to variable:", ast.gen_decl())
        self.ast = ast
        self.name = ast.name

        # format for struct
        fmt_var = self.fmtdict

        # Treat similar to class
        #        fmt_struct.class_scope = self.name + '::'
        fmt_var.field_name = ast.get_name(use_attr=False)
        fmt_var.variable_name = ast.name
        fmt_var.variable_lower = fmt_var.variable_name.lower()
        fmt_var.variable_upper = fmt_var.variable_name.upper()

        # Add to namespace


#        self.scope = self.parent.scope + self.name + '::'
#        self.typemap = typemap.create_struct_typemap(self)

######################################################################


class TemplateArgument(object):
    """Information used to instantiate a template.

    instantiation = "<int,double>"
    asts = [ Declaration("int"), Declaration("double") ]
    """

    def __init__(self, instantiation, fmtdict=None, options=None):
        self.instantiation = instantiation
        self.fmtdict = fmtdict
        self.options = options
        self.asts = None

    def parse_instantiation(self, namespace):
        """Parse instantiation (ex. <int>) and set list of Declarations."""
        parser = declast.Parser(self.instantiation, namespace)
        self.asts = parser.template_argument_list()


######################################################################


def create_std_namespace(glb):
    """Create the std namespace and add the types we care about.
    (string and vector)
    """
    std = glb.add_namespace("std")
    std.add_typedef("string")
    std.add_typedef("vector")
    return std


######################################################################
# Parse yaml file
######################################################################


def clean_dictionary(ddct):
    """YAML converts some blank fields to None,
    but we want blank.
    """
    for key in ["cxx_header", "namespace"]:
        if key in ddct and ddct[key] is None:
            ddct[key] = ""

    if "default_arg_suffix" in ddct:
        default_arg_suffix = ddct["default_arg_suffix"]
        if not isinstance(default_arg_suffix, list):
            raise RuntimeError("default_arg_suffix must be a list")
        for i, value in enumerate(ddct["default_arg_suffix"]):
            if value is None:
                ddct["default_arg_suffix"][i] = ""

    if "format" in ddct:
        fmtdict = ddct["format"]
        for key in ["function_suffix"]:
            if key in fmtdict and fmtdict[key] is None:
                fmtdict[key] = ""

    if "cxx_template" in ddct:
        # Convert to list of TemplateArgument instances
        cxx_template = ddct["cxx_template"]
        if not isinstance(cxx_template, list):
            raise RuntimeError("cxx_template must be a list")
        newlst = []
        for dct in cxx_template:
            if not isinstance(dct, dict):
                raise RuntimeError(
                    "cxx_template must be a list of dictionaries"
                )
            if "instantiation" not in dct:
                raise RuntimeError(
                    "instantation must be defined for each dictionary in cxx_template"
                )
            newlst.append(
                TemplateArgument(
                    dct["instantiation"],
                    fmtdict=dct.get("format", None),
                    options=dct.get("options", None),
                )
            )
        ddct["cxx_template"] = newlst


def clean_list(lst):
    """Fix up blank lines in a YAML line
    copyright:
    -  line one
    -
    -  next line

    YAML sets copyright[1] as null, change to empty string
    """
    for i, line in enumerate(lst):
        if line is None:
            lst[i] = ""


def add_declarations(parent, node):
    if "declarations" not in node:
        return
    if not node["declarations"]:
        return

    for subnode in node["declarations"]:
        if "block" in subnode:
            dct = copy.copy(subnode)
            clean_dictionary(dct)
            blk = BlockNode(parent, **dct)
            add_declarations(blk, subnode)
        elif "decl" in subnode:
            # copy before clean to avoid changing input dict
            dct = copy.copy(subnode)
            clean_dictionary(dct)
            decl = dct["decl"]
            del dct["decl"]
            declnode = parent.add_declaration(decl, **dct)
            add_declarations(declnode, subnode)
        else:
            print(subnode)
            raise RuntimeError(
                "Expected 'block', 'class', 'decl', 'forward', 'namespace' "
                "or 'typedef' found '{}'".format(sorted(subnode.keys()))
            )


def create_library_from_dictionary(node):
    """Create a library and add classes and functions from node.
    Typically, node is defined via YAML.

    library: name
    classes:
    - name: Class1
    functions:
    - decl: void func1()

    Do some checking on the input.
    Every class must have a name.
    """

    if "copyright" in node:
        clean_list(node["copyright"])

    clean_dictionary(node)
    library = LibraryNode(**node)
    ns = library

    # Create default namespace
    if "namespace" in node:
        for name in node["namespace"].split():
            ns = ns.add_namespace(name)

    if "typemap" in node:
        # list of dictionaries
        for subnode in node["typemap"]:
            # Update fields for a type. For example, set cpp_if
            key = subnode["type"]
            fields = subnode["fields"]
            def_types = typemap.get_global_types()
            ntypemap = def_types.get(key, None)
            if ntypemap:
                ntypemap.update(fields)
            else:
                # Create new typemap
                base = fields.get("base", "")
                if base == "shadow":
                    typemap.create_class_typemap_from_fields(
                        key, fields, library
                    )
                else:
                    raise RuntimeError("base must be 'shadow'")

    add_declarations(ns, node)

    return library
