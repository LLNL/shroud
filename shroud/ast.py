# Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and
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

from . import error
from . import declast
from . import statements
from . import todict
from . import typemap
from . import util
from . import visitor
from .util import wformat


class WrapFlags(object):
    """Keep track of which languages to wrap.
    """
    def __init__(self, options):
        self.fortran = options.get("wrap_fortran", False)
        self.c = options.get("wrap_c", False)
        self.f_or_c = self.fortran or self.c
        self.lua = options.get("wrap_lua", False)
        self.python = options.get("wrap_python", False)
        self.signature_c = None
        self.signature_f = None

    def clear(self):
        self.fortran = False
        self.c = False
        self.f_or_c = False
        self.lua = False
        self.python = False

    def assign(self, fortran=False, c=False, lua=False, python=False):
        """Assign wrap flags to wrap.

        Used when generating new FunctionNodes as part of function
        overload, generic, default args.
        """
        self.fortran = fortran
        self.c = c
        self.f_or_c = self.fortran or self.c
        self.lua = lua
        self.python = python

    def accumulate(self, wrap):
        """Accumulate flags via OR operator.

        Parameters
        ----------
        wrap : WrapFlags
        """
        self.fortran = self.fortran or wrap.fortran
        self.c = self.c or wrap.c
        self.f_or_c = self.fortran or self.c
        self.lua = self.lua or wrap.lua
        self.python = self.python or wrap.python

    def __str__(self):
        """Show which flags are set."""
        flags = []
        if self.fortran:
            flags.append("fortran")
        if self.c:
            flags.append("c")
        if self.lua:
            flags.append("lua")
        if self.python:
            flags.append("python")
        aflags = ",".join(flags)
        return "WrapFlags({})".format(aflags)


class AstNode(object):
    is_class = False

    def eval_template(self, name, tname="", fmt=None):
        """If a format has not been explicitly set, set from template."""
        if fmt is None:
            fmt = self.fmtdict
        if not fmt.inlocal(name):
            tname = name + tname + "_template"
            setattr(fmt, name, util.wformat(self.options[tname], fmt))

    def reeval_template(self, name, tname="", fmt=None):
        """Always evaluate template."""
        if fmt is None:
            fmt = self.fmtdict
        tname = name + tname + "_template"
        setattr(fmt, name, util.wformat(self.options[tname], fmt))

    def set_fmt_default(self, name, value, fmt=None):
        """Set a fmt value unless already set."""
        if fmt is None:
            fmt = self.fmtdict
        if not fmt.inlocal(name):
            setattr(fmt, name, value)

    def get_LibraryNode(self):
        """Return top of AST tree."""
        return self.parent.get_LibraryNode()

    def get_language(self):
        """Return language of library: c or c++"""
        return self.get_LibraryNode().language

    def lookup_class(self, name):
        """Lookup a class/struct AstNode by name.
        """
        node = self.get_LibraryNode().class_map.get(name, None)
        return node
    
    def find_header(self):
        """Return most recent cxx_header.
        Return list of headers to preserve order.
        """
        if self.cxx_header:
            return self.cxx_header
        elif self.parent is not None:
            return self.parent.find_header()
        else:
            return []

    def may_have_args(self):
        # only FunctionNode may have args
        return False

    def qualified_lookup(self, name):
        """Look for symbols within this AstNode."""
        return self.ast.qualified_lookup(name)

    def unqualified_lookup(self, name):
        """Look for symbols within the Abstract Syntax Tree
        and its parents."""
        return self.ast.unqualified_lookup(name)

    def apply_API_option(self, name, case, optname):
        """Apply case option to name
        Used with options  C_API_case and F_API_case.
        """
        if case == 'preserve':
            return name
        elif case == 'lower':
            return name.lower()
        elif case == 'upper':
            return name.upper()
        elif case == 'underscore':
            return util.un_camel(name)
        else:
            raise error.ShroudError(
                "Unexpected value of option {}: {}"
                .format(optname, case))

    def apply_C_API_option(self, name):
        """Apply option.C_API_case to name"""
        case = self.options.C_API_case
        return self.apply_API_option(name, case, "C_API_case")

    def apply_F_API_option(self, name):
        """Apply option.F_API_case to name"""
        case = self.options.F_API_case
        return self.apply_API_option(name, case, "F_API_case")

    def apply_LUA_API_option(self, name):
        """Apply option.LUA_API_case to name"""
        case = self.options.LUA_API_case
        return self.apply_API_option(name, case, "LUA_API_case")

    def update_names(self):
        """Update C and Fortran names.
        Necessary after templates are instantiated which
        defines fmt.function_suffix.
        """
        raise NotImplementedError("update_names for {}".format(self.__class__.__name__))

######################################################################

class NamespaceMixin(object):
    def add_class(self, decl, ast=None, fields={},
                  base=[], template_parameters=None,
                  **kwargs):
        """Add a class.

        template_parameters - list names of template parameters.
             ex. template<typename T>  -> ['T']

        Args:
            decl - str declaration ex. 'class cname'
            ast  - ast.Node of decl
            base - list of tuples ('public|private|protected', qualified-name (aa:bb), ntypemap)
        """
        node = ClassNode(
            decl, self, ast, base, fields=fields,
            template_parameters=template_parameters,
            **kwargs
        )
        self.classes.append(node)
        return node

    def add_declaration(self, decl, fields={}, **kwargs):
        """parse decl and add corresponding node.
        decl - declaration

        Called while reading YAML file.

        kwargs -
           cxx_template -
        """
        # parse declaration to find out what it is.
        fullast = declast.check_decl(decl, self.symtab)
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
                # XXX - move arguments to caller
                fmtdict = kwargs.pop("format", {})
                options = kwargs.pop("options", {})
                splicer = kwargs.pop("splicer", {})
                node = self.add_typedef(decl, ast, fields, fmtdict, options, splicer)
            elif ast.enum_specifier:
                fmtdict = kwargs.pop("format", {})
                options = kwargs.pop("options", {})
                splicer = kwargs.pop("splicer", {})
                node = self.add_enum(decl, ast, fmtdict, options, splicer, **kwargs)
            elif ast.class_specifier:
                if isinstance(ast.class_specifier, declast.Struct):
                    node = self.add_struct(
                        decl, ast=ast,
                        fields=fields,
                        template_parameters=template_parameters,
                        **kwargs)
                elif isinstance(ast.class_specifier, declast.CXXClass):
                    # A Class may already be forward defined.
                    # If so, just return it.
                    nodes = [cls for cls in self.classes if cls.name == ast.class_specifier.name]
                    if not nodes:
                        node = self.add_class(
                            decl, ast, fields=fields,
                            base=ast.class_specifier.baseclass,
                            template_parameters=template_parameters,
                            **kwargs
                        )
                    else:
                        if len(nodes) != 1:
                            raise RuntimeError(
                                "internal: too many nodes with the same name {}"
                                .format(ast.name))
                        node = nodes[0]
                else:
                    # Class or Union
                    raise RuntimeError("internal: add_declaration non-struct")
            elif ast.declarator.params is not None:
                node = self.add_function(decl, ast=fullast, **kwargs)
            else:
                node = self.add_variable(decl, ast=ast, **kwargs)
        elif isinstance(ast, declast.Namespace):
            node = self.add_namespace(decl, ast, **kwargs)
        else:
            raise RuntimeError(
                "add_declaration: unknown ast type {} after parsing '{}'".format(
                    type(ast), decl
                )
            )
        return node

    def add_enum(self, decl, ast=None,
                    format={}, options={}, splicer={}, **kwargs):
        """Add an enumeration.

        Add as a type for C++ but not C.
        """
        node = EnumNode(decl, self, ast, format, options, splicer, **kwargs)
        self.enums.append(node)
        return node

    def add_function(self, decl, ast=None, **kwargs):
        """Add a function.

        decl - C/C++ declaration of function
        ast  - parsed declaration. None if not yet parsed.
        """
        fcnnode = FunctionNode(decl, parent=self, ast=ast, **kwargs)
        self.functions.append(fcnnode)
        return fcnnode

    def add_namespace(self, decl, ast=None, expose=True, **kwargs):
        """Add a namespace.

        Args:
            decl - str declaration ex. 'namespace name'
            ast - declast.Node.  None for non-parsed namescopes like std.
            expose - If True, will be wrapped.
                     Otherwise, only used for lookup while parsing.
        """
        node = NamespaceNode(decl, parent=self, ast=ast, **kwargs)
        if not node.options.flatten_namespace and expose:
            self.namespaces.append(node)
        return node

    def add_struct(self, decl, ast=None, fields={},
                   template_parameters=None, **kwargs):
        """Add a struct.

        A struct is exactly like a class to the C++ compiler.  From
        the YAML, a struct may be a single ast or broken into parts.

        - decl: struct Cstruct1 {
                  int ifield;
                  double dfield;
                };
        - decl: struct Cstruct_ptr
          declarations:
          - decl: char *cfield;
          - decl: const double *const_dvalue;
        """
        if ast is None:
            ast = declast.check_decl(decl, self.symtab)
        class_specifier = ast.class_specifier
        name = class_specifier.name
        # XXX - base=... for inheritance
        node = ClassNode(decl, self, parse_keyword="struct",
                         ast=ast, fields=fields,
                         template_parameters=template_parameters,
                         **kwargs)
        for member in class_specifier.members:
            node.add_variable(str(member), member)
        self.classes.append(node)
        return node

    def add_typedef(self, decl, ast=None, fields={},
                    format={}, options={}, splicer={}):
        """Add a TypedefNode to the typedefs list.

        This may be the YAML file as a typemap which may have 'fields',
        or a decl with a typemap.
        """
        if ast is None:
            ast = declast.check_decl(decl, self.symtab)

        name = ast.declarator.user_name  # Local name.
        node = TypedefNode(name, self, ast, fields, format, options, splicer)
        self.typedefs.append(node)

        # Add typedefs names to structs/enums.
        # Check if typedef is for a struct tag
        # find matching struct node
        # update name in typemap
        if ast.class_specifier:
            struct = ast.class_specifier
            for cls in self.classes:
                if cls.ast.class_specifier is struct:
                    cls.rename(node)
                    typemap.fill_struct_typemap(cls, cls.user_fields)
        
        return node

    def add_variable(self, decl, ast=None, **kwargs):
        """Add a variable or class member.

        Create a VariableNode for each declarator.

        decl - C/C++ declaration of function
        ast  - parsed declaration. None if not yet parsed.
        """
        node = VariableNode(decl, parent=self, ast=ast, **kwargs)
        self.variables.append(node)
        if len(ast.declarators) > 1:
            declarators = ast.declarators
            ast.declarators = []
            for declarator in declarators[1:]:
                d2 = copy.copy(ast)
                d2.declarator = declarator
                node = VariableNode(decl, parent=self, ast=d2, **kwargs)
                self.variables.append(node)
        return node


######################################################################


class LibraryNode(AstNode, NamespaceMixin):
    """There is one library per wrapping.
    It represents the global namespace.
    """
    def __init__(
        self,
        symtab=None,
        cxx_header="",
        fortran_header="",
        namespace=None,
        format={},
        language="c++",
        library="library",
        options=None,
        **kwargs
    ):
        """Create LibraryNode.
        Represents the global namespace.

        cxx_header = blank delimited list of headers for C++ or C library.

        Args:
            namespace - blank delimited list of initial namespaces.

        fields = value
        options:
        classes:
        functions:

        wrap_namespace - Node to start wrapping.  This is the current node but 
            will be changed if the top level "namespace" variable is set.
        """
        # From arguments
        self.parent = None
        self.cxx_header = cxx_header.split()
        self.fortran_header = fortran_header.split()
        self.language = util.find_language(language)
        self.library = library
        self.name = library
        self.nodename = "library"
        self.wrap_namespace = self

        self.classes = []
        self.enums = []
        self.functions = []
        self.namespaces = []
        self.typedefs = []
        self.variables = []
        # Each is given a _function_index when created.
        self.function_index = []
        self.class_map = {}   # indexed by Typemap.flat_name, holds AST (not typemap).
        # Headers required by template arguments.
        self.gen_headers_typedef = {}

        # namespace
        self.scope = ""
        self.scope_file = [library]

        self.options = self.default_options()
        if options:
            self.options.update(options, replace=True)
        self.wrap = WrapFlags(self.options)
        if self.options.literalinclude:
            # global literalinclude implies literalinclude2
            self.options.literalinclude2 = True

        self.F_module_dependencies = []  # unused

        self.copyright = kwargs.get("copyright", [])
        self.patterns = kwargs.get("patterns", [])

        # Convert file_code into typemaps to use in class util.Headers.
        # This feels like a kludge and should be refined.
        self.file_code = {}
        if "file_code" in kwargs:
            for fname, values in kwargs["file_code"].items():
                if fname == "__line__":
                    continue
                ntypemap = typemap.Typemap(fname)
                ntypemap.update(values)
                self.file_code[fname] = ntypemap

        self.user_fmt = format
        self.default_format(format, kwargs)

        # Create a symbol table and a 'fake' AST node for global.
        self.symtab = symtab or declast.SymbolTable()
        self.symtab.create_std_names()  # size_t et al.

        if self.language == "cxx":
            self.symtab.create_std_namespace()
            self.symtab.using_directive("std")

        # Create default namespace
        if namespace:
            ns = self
            for name in namespace.split():
                ns = ns.add_namespace("namespace " + name, skip=True)
            # Any namespaces listed in the "namespace" field are not wrapped.
            self.wrap_namespace = ns

        self.ast = self.symtab.current  # declast.Global

        self.setup = kwargs.get("setup", {}) # for setup.py

    def get_LibraryNode(self):
        """Return top of AST tree."""
        return self

    #####

    def default_options(self):
        """default options."""
        def_options = util.Scope(
            parent=None,
            debug=False,  # print additional debug info
            debug_index=False,  # print function indexes. debug must also be True.
            debug_testsuite=False,
            # They change when a function is inserted.
            flatten_namespace=False,
            wrap_class_as="class",
            wrap_struct_as="struct",
            class_baseclass=None,
            class_ctor=None,
            class_method=None,
            C_force_wrapper=False,
            C_line_length=72,
            F_CFI=False,    # TS29113 C Fortran Interoperability
            F_assumed_rank_min=0,
            F_assumed_rank_max=7,
            F_blanknull=False,
            F_create_bufferify_function=True,
            F_default_args="generic",  # "generic", "optional", "require"
            F_enum_type="int",
            F_flatten_namespace=False,
            F_line_length=72,
            F_force_wrapper=False,
            F_return_fortran_pointer=True,
            F_standard=2003,
            F_struct_getter_setter=True,
            F_trim_char_in=True,
            F_auto_reference_count=False,
            F_create_generic=True,
            # wrap options are set in by command line options in main.py
#            wrap_c=True,
#            wrap_fortran=True,
#            wrap_python=False,
#            wrap_lua=False,
            doxygen=True,  # create doxygen comments
            literalinclude=False, # Create sphinx literalinclude markers
            literalinclude2=False, # Used with global identifiers
            return_scalar_pointer="pointer",
            show_splicer_comments=True,
            # blank for functions, set in classes.
            YAML_type_filename_template="{library_lower}_types.yaml",

            C_API_case="preserve",
            C_header_filename_library_template="wrap{library}.{C_header_filename_suffix}",
            C_impl_filename_library_template="wrap{library}.{C_impl_filename_suffix}",

            C_header_filename_namespace_template="wrap{file_scope}.{C_header_filename_suffix}",
            C_impl_filename_namespace_template="wrap{file_scope}.{C_impl_filename_suffix}",

            C_header_filename_class_template="wrap{file_scope}.{C_header_filename_suffix}",
            C_impl_filename_class_template="wrap{file_scope}.{C_impl_filename_suffix}",

            C_header_utility_template="types{library}.{C_header_filename_suffix}",
            C_impl_utility_template="util{library}.{C_impl_filename_suffix}",
            C_enum_type_template="{C_prefix}{C_name_scope}{enum_name}",
            C_enum_member_template="{C_prefix}{C_name_scope}{C_name_api}",
            C_name_template=(
                "{C_prefix}{C_name_scope}{C_name_api}{function_suffix}{f_c_suffix}{template_suffix}"
            ),
            C_name_typedef_template="{C_prefix}{C_name_scope}{typedef_name}",
            C_memory_dtor_function_template=(
                "{C_prefix}SHROUD_memory_destructor"
            ),
            C_shadow_result=True,               # Return pointer to capsule
            C_var_capsule_template="C{c_var}",  # capsule argument
            C_var_context_template="D{c_var}",  # context argument
#            C_var_len_template="N{c_var}",  # argument for result of len(arg)
#            C_var_trim_template="L{c_var}",  # argument for result of len_trim(arg)
#            C_var_size_template="S{c_var}",  # argument for result of size(arg)
            CXX_standard=2011,
            # Fortran's names for C functions
            F_API_case="underscore",
            F_C_name_template=(
                "{F_C_prefix}{F_name_scope}{F_name_api}{function_suffix}{f_c_suffix}{template_suffix}"
            ),
            F_enum_member_template="{F_name_scope}{F_name_api}",
            F_name_impl_template=(
                "{F_name_scope}{F_name_api}{function_suffix}{template_suffix}"
            ),
            F_name_function_template="{F_name_api}{function_suffix}{template_suffix}",
            F_name_generic_template="{F_name_api}",
            F_name_enum_template="{F_name_scope}{F_name_api}",
            F_name_typedef_template="{F_name_scope}{F_name_api}",
            F_module_name_library_template="{library_lower}_mod",
            F_module_name_namespace_template="{file_scope}_mod",
            F_impl_filename_library_template="wrapf{library_lower}.{F_filename_suffix}",
            F_impl_filename_namespace_template="wrapf{file_scope}.{F_filename_suffix}",
            F_array_type_template="{C_prefix}SHROUD_array",
            F_capsule_data_type_template="{C_prefix}SHROUD_capsule_data",
            F_capsule_type_template="{C_prefix}SHROUD_capsule",
            F_abstract_interface_subprogram_template="{F_name_api}_{argname}",
            F_abstract_interface_argument_template="arg{index}",
            F_derived_name_template="{F_name_api}",

            LUA_API_case="preserve",
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
            LUA_name_impl_template="{LUA_prefix}{C_name_scope}{LUA_name_api}",

            PY_create_generic=True,
            PY_module_filename_template=(
                "py{file_scope}module.{PY_impl_filename_suffix}"
            ),
            PY_header_filename_template=(
                "py{library}module.{PY_header_filename_suffix}"
            ),
            PY_utility_filename_template=(
                "py{library}util.{PY_impl_filename_suffix}"
            ),
            PY_write_helper_in_util=False,
            PY_PyTypeObject_template="{PY_prefix}{cxx_class}_Type",
            PY_PyObject_template="{PY_prefix}{cxx_class}",
            PY_type_filename_template=(
                "py{file_scope}type.{PY_impl_filename_suffix}"
            ),
            PY_name_impl_template=(
                "{PY_prefix}{function_name}{function_suffix}{template_suffix}"
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
            PY_member_object_template="{variable_name}_obj",
            PY_member_data_template="{variable_name}_dataobj",
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

            # Functions created by Shroud
            SH_class_getter_template="get_{wrapped_name}",
            SH_class_setter_template="set_{wrapped_name}",
            SH_struct_getter_template="{struct_name}_get_{wrapped_name}",
            SH_struct_setter_template="{struct_name}_set_{wrapped_name}",

            typemap_sgroup=None,
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
            C_cfi_suffix="_CFI",
            C_call_list="",
            C_prefix=C_prefix,
            C_result="rv",  # return value
            c_temp="SHT_",
            C_local="SHC_",
            C_name_scope = "",
            C_name_typedef="",
            C_this="self",
            C_custom_return_type="",  # assume no value
            CXX_this="SH_this",
            CXX_local="SHCXX_",
            cxx_class="",  # Assume no class
            class_scope="",
            file_scope = "_".join(self.scope_file),
            F_arg_c_call="",
            F_C_prefix="c_",
            F_C_name="-F_C_name-",
            F_derived_member="cxxmem",
            F_derived_member_base="",
            F_name_assign="assign",
            F_name_associated="associated",
            F_name_enum="",
            F_name_instance_get="get_instance",
            F_name_instance_set="set_instance",
            F_name_final="final",
            F_name_typedef="",
            F_result="SHT_rv",
            F_result_ptr="SHT_prv",
            F_name_scope = "",
            F_this="obj",
            C_string_result_as_arg="SHF_rv",
            F_string_result_as_arg="",
            F_capsule_final_function="SHROUD_capsule_final",
            F_capsule_delete_function="SHROUD_capsule_delete",

            c_blanknull="0",     # Argument to helper ShroudStrAlloc.
            c_array_shape="",
            c_array_size="1",
            c_char_len="0",      # deferred length
            # Assume scalar in CFI_establish
            c_temp_extents_decl="",
            c_temp_extents_use="NULL",
            # Assume scalar in CFI_setpointer
            c_temp_lower_decl="",    # Assume scalar.
            c_temp_lower_use="NULL",   # Assume scalar in CFI_setpointer.

            f_abstract_interface="",
            f_array_allocate="",
            f_array_shape="",
            f_assumed_shape="",  # scalar
            f_char_len=":",      # deferred length
            f_char_type="",      # allocate type - character(len=x) ::
            f_declare_shape_prefix="SHAPE_",
            f_declare_shape_array="",
            f_get_shape_array="",
            f_dimension="",
            f_intent="",
            f_intent_attr="",
            f_kind="",
            f_optional_attr="",
            f_shape_var="",
            f_type="",
            f_value_attr="",
            f_var_shape="",      # scalar

            # Fortran interface
            i_dimension="",
            
            rank="0",            # scalar
            
            LUA_result="rv",
            LUA_prefix="l_",
            LUA_state_var="L",
            LUA_this_call="",

            PY_ARRAY_UNIQUE_SYMBOL="SHROUD_{}_ARRAY_API".format(
                self.library.upper()),
            PY_helper_prefix="SHROUD_",
            PY_prefix="PY_",
            PY_module_name=self.library.lower(),
            PY_result="SHTPy_rv",  # Create PyObject for result
            PY_this_call="",
            PY_type_obj="obj",  # name of cpp class pointer in PyObject
            PY_type_dtor="idtor",  # name of destructor capsule infomation
            PY_value_init="{NULL, NULL, NULL, NULL, 0}",  # initial value for PY_typedef_converter

            library=self.library,
            library_lower=self.library.lower(),
            library_upper=self.library.upper(),
            # set default values for fields which may be unset.
            # c_const='',
            CXX_this_call="",
            CXX_template="",
            C_pre_call="",
            C_post_call="",
            function_suffix="",  # assume no suffix
            f_c_suffix="",
            template_suffix="",  # assume no suffix
            namespace_scope="",
        )

        if False:
            # Add default values to format to aid debugging.
            # Avoids exception from wformat for non-existent fields.
            fmt_library.update(dict(
                c_get_value="XXXc_get_value",
                c_type="XXXc_type",
                c_var="XXXc_var",
                c_var_capsule="XXXc_var_capsule",
                c_var_cdesc="XXXc_var_cdesc",
                c_var_dimension="XXXc_var_dimension",
                c_var_len="XXXc_var_len",
                c_var_trim="XXXc_var_trim",
                i_dimension="XXXi_dimension",
                i_module_line="XXXi_module_line:XXXnone",
                i_type="XXXi_type",
                cxx_addr="XXXcxx_addr",
                cxx_member="XXXcxx_member",
                cxx_nonconst_ptr="XXXcxx_nonconst_ptr",
                cxx_type="XXXcxx_type",
                cxx_var="XXXcxx_var",
#                cxx_T="short",   # Needs to be a actual type to find helper.
                F_C_var="XXXF_C_var",
                f_abstract_interface="XXXf_abstract_interface=",
                f_capsule_data_type="XXXf_capsule_data_type",
                f_cdesc_shape="XXXf_cdesc_shape",
                f_intent="XXXf_intent",
                f_type="XXXf_type",
                f_var="XXXf_var",
                idtor="XXXidtor",
                PY_member_object="XXXPY_member_object",
                PY_to_object_func="XXXPY_to_object_func",
                temp0="XXXtemp0",
                temp1="XXXtemp1",
                temp2="XXXtemp2",
            ))

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
            fmt_library.nullptr = "NULL"
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
            if self.options.CXX_standard >= 2011:
                fmt_library.nullptr = "nullptr"
            else:
                fmt_library.nullptr = "NULL"

        # Update format based on options
        options = self.options
        if options.PY_write_helper_in_util:
            fmt_library.PY_helper_static = ""
            fmt_library.PY_helper_prefix = (
                fmt_library.C_prefix + fmt_library.PY_helper_prefix )
        else:        
            fmt_library.PY_helper_static = "static "
        fmt_library.PY_typedef_converter = (
                fmt_library.C_prefix + "SHROUD_converter_value")

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

        if fmtdict:
            fmt_library.update(fmtdict, replace=True)

        self.fmtdict = fmt_library

        # default some format strings based on other format strings
        self.set_fmt_default("C_array_type",
                             fmt_library.C_prefix + "SHROUD_array")
        self.set_fmt_default("C_capsule_data_type",
                             fmt_library.C_prefix + "SHROUD_capsule_data")

        self.eval_template("C_header_filename", "_library")
        self.eval_template("C_impl_filename", "_library")
        self.eval_template("C_header_utility")
        self.eval_template("C_impl_utility")

        self.eval_template("C_memory_dtor_function")

        self.eval_template("F_array_type")
        self.eval_template("F_capsule_type")
        # All class/methods and functions may go into this file or
        # just functions.
        self.eval_template("F_module_name", "_library")
        fmt_library.F_module_name = fmt_library.F_module_name.lower()
        self.eval_template("F_impl_filename", "_library")
        self.eval_template("F_capsule_data_type")

        # If user changes PY_module_name, reflect change in PY_module_scope.
        self.set_fmt_default(
            "PY_module_init", fmt_library.PY_module_name, fmt_library)
        self.set_fmt_default(
            "PY_module_scope", fmt_library.PY_module_name, fmt_library)
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

    def __init__(self, parent, format={}, options=None, **kwargs):
        # From arguments
        self.parent = parent
        self.symtab = parent.symtab

        self.classes = parent.classes
        self.enums = parent.enums
        self.functions = parent.functions
        self.namespaces = parent.namespaces
        self.typedefs = parent.typedefs
        self.variables = parent.variables
        self.scope = parent.scope
        self.scope_file = parent.scope_file
        self.cxx_header = parent.cxx_header

        self.options = util.Scope(parent=parent.options)
        if options:
            self.options.update(options, replace=True)

        self.user_fmt = format
        self.fmtdict = util.Scope(parent=parent.fmtdict)
        if format:
            self.fmtdict.update(format, replace=True)


######################################################################


class NamespaceNode(AstNode, NamespaceMixin):
    def __init__(self, decl, parent, ast=None, cxx_header="",
                 format={}, options=None, skip=False, **kwargs):
        """Create NamespaceNode.

        parent may be LibraryNode or NamespaceNode.

        Args:
            skip - skip when generating scope_file and format names since
                   it is part of the initial namespace, not a namespace
                   within a declaration.
        """
        # From arguments
        self.parent = parent
        self.symtab = parent.symtab
        self.cxx_header = cxx_header.split()
        self.nodename = "namespace"
        self.linenumber = kwargs.get("__line__", "?")

        if not decl:
            raise RuntimeError("NamespaceNode missing decl");
        self.decl = decl
        if ast is None:
            ast = declast.check_decl(decl, parent.symtab)
        if not isinstance(ast, declast.Namespace):
            raise RuntimeError("namespace decl is not a Namespace Node")
        self.ast = ast
        self.name = ast.name

        self.options = util.Scope(parent=parent.options)
        if options:
            self.options.update(options, replace=True)
        self.wrap = WrapFlags(self.options)
        self.file_code = {}     # Only used for LibraryNode.

        if self.options.flatten_namespace:
            self.classes = parent.classes
            self.enums = parent.enums
            self.functions = parent.functions
            self.namespaces = parent.namespaces
            self.typedefs = parent.typedefs
            self.variables = parent.variables
        else:
            self.classes = []
            self.enums = []
            self.functions = []
            self.namespaces = []
            self.typedefs = []
            self.variables = []

        # Headers required by template arguments.
        self.gen_headers_typedef = {}

        # Create scope
        self.scope = self.parent.scope + self.name + "::"
        if skip:
            self.scope_file = self.parent.scope_file
        else:
            self.scope_file = self.parent.scope_file + [self.name]

        self.user_fmt = format
        self.default_format(parent, skip)

    #####

    def default_format(self, parent, skip=False):
        """Set format dictionary.

        Parameters
        ----------
        skip : Skip adding to name_scope.
        """

        options = self.options
        self.fmtdict = util.Scope(parent=parent.fmtdict)

        fmt_ns = self.fmtdict
        fmt_ns.namespace_scope = (
            parent.fmtdict.namespace_scope + self.name + "::"
        )
        fmt_ns.C_name_api = self.apply_C_API_option(self.name)
        fmt_ns.F_name_api = self.apply_F_API_option(self.name)

        if self.user_fmt:
            fmt_ns.update(self.user_fmt, replace=True)

        if not skip:
            if fmt_ns.C_name_api:
                fmt_ns.C_name_scope = (
                    parent.fmtdict.C_name_scope + fmt_ns.C_name_api + "_"
                )
            if fmt_ns.F_name_api:
                if options.flatten_namespace or options.F_flatten_namespace:
                    fmt_ns.F_name_scope = (
                        parent.fmtdict.F_name_scope + fmt_ns.F_name_api + "_"
                    )
        fmt_ns.file_scope = "_".join(self.scope_file)
        fmt_ns.CXX_this_call = fmt_ns.namespace_scope
        fmt_ns.LUA_this_call = fmt_ns.namespace_scope
        fmt_ns.PY_this_call = fmt_ns.namespace_scope
        if not skip:
            fmt_ns.PY_module_name = self.name

        self.eval_template("C_header_filename", "_namespace")
        self.eval_template("C_impl_filename", "_namespace")
        if skip:
            # No module will be created for this namespace, use library template.
            self.eval_template("F_impl_filename", "_library")
            self.eval_template("F_module_name", "_library")
        else:
            self.eval_template("F_impl_filename", "_namespace")
            self.eval_template("F_module_name", "_namespace")
        fmt_ns.F_module_name = fmt_ns.F_module_name.lower()

        # If user changes PY_module_name, reflect change in PY_module_scope.
        if not skip:
            self.set_fmt_default(
                "PY_module_init",
                parent.fmtdict.PY_module_init + "_" + fmt_ns.PY_module_name,
                fmt_ns
            )
            self.set_fmt_default(
                "PY_module_scope",
                parent.fmtdict.PY_module_scope + "." + fmt_ns.PY_module_name,
                fmt_ns
            )


######################################################################


class ClassNode(AstNode, NamespaceMixin):
    """A C++ class or struct.

    """

    is_class = True

    def __init__(
        self,
        decl,
        parent,
        ast=None,
        base=[],
        cxx_header="",
        format={},
        fields={},
        options=None,
        parse_keyword="class",
        template_parameters=None,
        **kwargs
    ):
        """Create ClassNode.
        Used with classes and structs.

        template_parameters - list names of template parameters.
             ex. template<typename T>  -> ['T']

        cxx_template - list of TemplateArgument instances

        Args:
            base - list of tuples ('public|private|protected', qualified-name (aa:bb), ntypemap)
            parse_keyword - keyword from decl - "class" or "struct".
        """
        # From arguments
        self.parent = parent
        self.symtab = parent.symtab
        self.baseclass = base
        self.parse_keyword = parse_keyword
        self.cxx_header = cxx_header.split()
        self.nodename = "class"
        self.linenumber = kwargs.get("__line__", "?")

        self.classes = []
        self.enums = []
        self.functions = []
        self.typedefs = []
        self.variables = []

        self.typedef_map = []

        self.python = kwargs.get("python", {})
        self.cpp_if = kwargs.get("cpp_if", None)

        self.options = util.Scope(parent=parent.options)
        if options:
            self.options.update(options, replace=True)
        self.wrap = WrapFlags(self.options)

        if not decl:
            raise RuntimeError("ClassNode missing decl");
        self.decl = decl
        if ast is None:
            ast = declast.check_decl(decl, parent.symtab)
        if not isinstance(ast, declast.Declaration):
            raise RuntimeError("class decl is not a Declaration")
        class_specifier = ast.class_specifier
        self.ast = ast
        if not isinstance(class_specifier, (declast.CXXClass, declast.Struct)):
            raise RuntimeError("class decl is not a CXXClass or Struct Node")
        self.name = class_specifier.name
        self.name_api = None            # ex. name_int
        self.name_instantiation = None  # ex. name<int>

        self.scope = self.parent.scope + self.name + "::"
        self.scope_file = self.parent.scope_file + [self.name]

        self.user_fmt = format
        self.deprecated_format(kwargs)
        self.fmtdict = util.Scope(parent.fmtdict)
        self.default_format()

        if self.parse_keyword == "struct":
            self.wrap_as = self.options.wrap_struct_as
        elif self.parse_keyword == "class":
            self.wrap_as = self.options.wrap_class_as
        else:
            raise TypeError("parse_keyword must be 'class' or 'struct'")

        self.user_fields = fields
        self.typemap = ast.typemap
        if self.wrap_as == "struct":
            typemap.fill_struct_typemap(self, fields)
        elif self.wrap_as == "class":
            typemap.fill_class_typemap(self, fields)

        if self.options.typemap_sgroup is not None:
            # Control selection of fc-statements.
            self.typemap.sgroup = self.options.typemap_sgroup

        if format and 'template_suffix' in format:
            # Do not use scope from self.fmtdict, instead only copy value
            # when in the format dictionary is passed in.
            self.typemap.template_suffix = format['template_suffix']

        # Add template parameters.
        if template_parameters is None:
            self.template_parameters = []
        else:
            self.template_parameters = template_parameters  # GGG _name

        # Parse the instantiations.
        # cxx_template = [ TemplateArgument('<int>'),
        #                  TemplateArgument('<double>') ]
        cxx_template = kwargs.get("cxx_template", [])
        self.template_arguments = cxx_template
        for args in cxx_template:
            args.parse_instantiation(self.symtab)
        # Headers required by template arguments.
        self.gen_headers_typedef = {}

    #####

    def deprecated_format(self, kwargs):
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

    def default_format(self):
        """Set format dictionary for a ClassNode."""
        name_api = self.name_api or self.name
        name_instantiation = self.name_instantiation or self.name

        self.fmtdict.update(dict(
            cxx_type=name_instantiation,
            cxx_class=name_api,

            C_name_api = self.apply_C_API_option(name_api),
            F_name_api = self.apply_F_API_option(name_api),

            class_scope=name_instantiation + "::",
#            namespace_scope=self.parent.fmtdict.namespace_scope + name_api + "::",

            # The scope for things in the class.
            file_scope="_".join(self.scope_file[1:]),
        ))

        if self.user_fmt:
            self.fmtdict.update(self.user_fmt, replace=True)
        self.expand_format_templates()

    def expand_format_templates(self):
        """Expand format templates for a class.
        Called after other format fields are set.
        eval_template will only set a value if it has not already been set.
        Call delete_format_template to remove previous values to
        force them to be recomputed during class template instantiation.
        """
        # Only one file per class for C.
        self.eval_template("C_header_filename", "_class")
        self.eval_template("C_impl_filename", "_class")

        self.eval_template("F_derived_name")

        fmt = self.fmtdict
        fmt.C_name_scope = self.parent.fmtdict.C_name_scope + fmt.C_name_api + "_"
        fmt.F_name_scope = self.parent.fmtdict.F_name_scope + fmt.F_name_api + "_"
        
        # As PyArray_Descr
        if self.parse_keyword == "struct":
            self.eval_template("PY_struct_array_descr_create")
            self.eval_template("PY_struct_array_descr_variable")
            self.eval_template("PY_struct_array_descr_name")

    def delete_format_templates(self):
        """Delete some format strings which were defaulted.
        Used when instantiation a class to remove previous class' values.
        """
        self.fmtdict.delattrs(
            [
                "C_header_filename",
                "C_impl_filename",
                "F_derived_name",
                "F_module_name",
                "F_impl_filename",
                "F_capsule_data_type",
                "PY_struct_array_descr_create",
                "PY_struct_array_descr_variable",
                "PY_struct_array_descr_name",
            ]
        )

    def rename(self, nameobj):
        """Rename the class.

        Used with templated classes and typedef which
        can change the name of a class/struct.

        nameobj - TemplateArgument,  TypedefNode
        """
        self.name = nameobj.name
        # XXX - what about templated typedefs (using statement)
        self.delete_format_templates()
        self.default_format()
        
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
        new.wrap = WrapFlags(self.options)
        new.scope_file = self.scope_file[:]
        new.user_fmt = self.user_fmt.copy()

        # Clone all functions.
        newfcns = []
        for fcn in self.functions:
            newfcn = fcn.clone()
            newfcn.fmtdict.reparent(new.fmtdict)
            newfcn.options.reparent(new.options)
            newfcns.append(newfcn)
        new.functions = newfcns

        # Clone all typedefs
        newtyps = []
        for typ in self.typedefs:
            newtyp = typ.clone()
            newtyp.fmtdict.reparent(new.fmtdict)
            newtyp.options.reparent(new.options)
            newtyps.append(newtyp)
        new.typedefs = newtyps

        return new

    def create_node_map(self):
        """Create a map from the name to Node for declarations in 
        the class.
        """
        self.map_name_to_node = {}
        for var in self.variables:
            self.map_name_to_node[var.name] = var
        for node in self.functions:
            self.map_name_to_node[node.ast.declarator.user_name] = node

    def qualified_lookup(self, name):
        """Look for symbols within this AstNode."""
        return self.ast.class_specifier.qualified_lookup(name)

    def unqualified_lookup(self, name):
        """Look for symbols within the Abstract Syntax Tree
        and its parents."""
        return self.ast.class_specifier.unqualified_lookup(name)


######################################################################


class FunctionNode(AstNode):
    """

    - decl: template<typename T1, typename T2> foo(T1 arg1, T2 arg2)
      cxx_template:
      - instantiation: <int, long>
      - instantiation: <float, double>
      fortran_generic:
      - decl: (float arg)
      - decl: (double arg)
      fattrs:     # function attributes
      attrs:
        arg1:     # argument attributes
     bind:
        f:
          decl:   (arg1 +attr, arg2+attr)  +attr
          attrs:
             +result:
               attr2: true
             arg1:
      splicer:
         c: [ ]
         f: [ ]
         py: [ ]
      fstatements: # function statements
         c:
         f:
         py:


    _fmtfunc = Scope()
    _fmtlang = {
       'f': Scope(_fmtfunc)
    }

    _bind = {
       'f': {
             '+result': { }
             'arg1': {    statements.BindArg
                stmt: Scope
                meta: collections.defaultdict(lambda: None)
                fmtdict:  Scope(_fmtfunc)
              }
        },
        'c': {},
        'lua': {},
        'py': {},
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
    _generated       - which method generated this function
    _generated_path  - list of generated functions
    _orig_node       - Original YAML node for generated functions.
    _PTR_F_C_index   - Used by fortran wrapper to find index of
                       C function to call
    _PTR_C_CXX_index - Used by C wrapper to find index of C++ function
                       to call

    Templates
    ---------

    template_parameters - [ 'T1', 'T2' ]
    template_argument - [ TemplateArgument('<int,long>'),
                          TemplateArgument('<float,double>') ]

    fortran_generic = [ FortranGeneric('double arg'),
                        FortranGeneric('float arg') ]

    A list of helpers for the function derived from statements.
    helpers = {
      'c': {
            name = True,
           }
      'f': {},
    }

    C_signature - statement.index signature of C wrapper.
       Used to avoid writing the same function twice. 
    C_fortran_generic = if True, multiple version of the Fortran
       wrappers will call a single C wrapper.
    """

    def __init__(
        self, decl, parent, format={}, ast=None, options=None, **kwargs
    ):
        """
        ast - None, declast.Declaration, declast.Template
        """
        self.parent = parent
        self.symtab = parent.symtab
        self.linenumber = kwargs.get("__line__", "?")

        self.options = util.Scope(parent.options)
        if options:
            self.options.update(options, replace=True)
        self.wrap = WrapFlags(self.options)

        # working variables
        self._PTR_C_CXX_index = None
        self._PTR_F_C_index = None
        self.cxx_header = []
        self._cxx_overload = None
        self.declgen = None  # generated declaration.
        self._default_funcs = []  # generated default value functions  (unused?)
        self._fmtlang = {}
        self._function_index = None
        self._generated = False
        self._generated_path = []
        self._orig_node = None
        self._has_default_arg = False
        self._nargs = None
        self._overloaded = False
        self._gen_fortran_generic = False # An argument is assumed-rank.
        self._bind = {}                   # Access with get_func_bind or get_arg_bind
        self.splicer = {}
        self.fstatements = {}
        self.struct_parent = None         # Function is a getter/setter for a struct
        self.struct_members = {}          # VariableNode for class ctor

        # Fortran wapper variables.
        self.C_node = None   # C wrapper required by Fortran wrapper
        self.C_force_wrapper = False
        self.C_signature = None
        self.C_fortran_generic = False

        # self.function_index = []

        self.default_arg_suffix = kwargs.get("default_arg_suffix", [])
        self.cpp_if = kwargs.get("cpp_if", None)
        self.cxx_template = {}
        self.template_parameters = []
        self.template_arguments = kwargs.get("cxx_template", [])
        self.doxygen = kwargs.get("doxygen", {})
        self.fortran_generic = kwargs.get("fortran_generic", [])
        self.return_this = kwargs.get("return_this", False)
        self.helpers = {}

        # Headers required by template arguments.
        self.gen_headers_typedef = {}
        error.cursor.push_node(self)

        if not decl:
            raise RuntimeError("FunctionNode missing decl")

        self.decl = decl
        if ast is None:
            ast = declast.check_decl(decl, parent.symtab)
            
        if isinstance(ast, declast.Template):
            self.name = ast.decl.declarator.name
            if not self.template_arguments:
                error.cursor.warning("Templated function requires the 'cxx_template' field")
            for param in ast.parameters:
                self.template_parameters.append(param.name)

            template_parameters = ast
            ast = ast.decl
            for args in self.template_arguments:
                args.parse_instantiation(self.symtab)

            # XXX - convert to cxx_template format  { T=['int', 'double'] }
            # XXX - only deals with single template argument  [0]?
            argname = template_parameters.parameters[0].name
            lst = []
            for arg in self.template_arguments:
                lst.append(arg.asts[0].typemap.name)
            self.cxx_template[argname] = lst
        elif isinstance(ast, declast.Declaration):
            self.name = ast.declarator.name
            if self.template_arguments:
                error.cursor.warning("'cxx_template' field only used with a templated function")
        else:
            raise RuntimeError("Expected a function declaration")
        if ast.declarator.params is None:
            # 'void foo' instead of 'void foo()'
            raise RuntimeError("Missing arguments to function:", ast.gen_decl())
        self.ast = ast
        declarator = ast.declarator
        self.name = declarator.user_name
        self.user_fmt = format
        self.default_format(parent, format, kwargs)

        if self.return_this:
            if not isinstance(self.parent, ClassNode):
                raise error.ShroudError(
                    "return_this can only be set for class methods")
            if self.parent.typemap is not ast.typemap:
                raise error.ShroudError(
                    "return_this must return a pointer to the class")
            if ast.declarator.is_pointer() != 1:
                raise error.ShroudError(
                    "return_this must return a pointer to the class")

        # Look for any template (include class template) arguments.
        self.have_template_args = False
        if ast.typemap.base == "template":
            self.have_template_args = True
        else:
            for args in declarator.params:
                if args.typemap.base == "template":
                    self.have_template_args = True
                    break

        # Compute full param list for each generic specification
        # by copying original params then substituting decls from fortran_generic.
        for generic in self.fortran_generic:
            generic.parse_generic(self.symtab)
            newparams = copy.deepcopy(declarator.params)
            first = len(newparams) + 1
            for garg in generic.decls:
                i = declast.find_arg_index_by_name(newparams, garg.declarator.user_name)
                if i < 0:
                    # XXX - For default argument, the generic argument may not exist.
                    error.cursor.ast(self.linenumber,
                                    "Error in 'fortran_generic', argument '{}' not found".format(
                                        garg.declarator.user_name))
                    self.wrap.clear()
                else:
                    first = min(first, i + 1)
                    newparams[i] = garg
            generic.decls = newparams
            generic.first = first

        # add any attributes from YAML files to the ast
        if "attrs" in kwargs:
            attrs = kwargs["attrs"]
            for arg in declarator.params:
                name = arg.declarator.user_name
                if name in attrs:
                    arg.declarator.attrs.update(attrs[name])
        if "fattrs" in kwargs:
            declarator.attrs.update(kwargs["fattrs"])

        if "splicer" in kwargs:
            self.splicer = kwargs["splicer"]
            
        if "fstatements" in kwargs:
            # fstatements must be a dict
            for key, value in kwargs["fstatements"].items():
                # value must be a dict
                if key in ["c", "c_buf", "f", "py"]:
                    # remove __line__?
                    self.fstatements[key] = util.Scope(None, **value)
        if "bind" in kwargs:
            # lang must be a dict
            for wlang, langdict in kwargs["bind"].items():
                # langdict must be a dict
                if wlang not in ["c", "f", "py"]:
                    continue  # error
                value = langdict.get("decl")
                if value:
                    bind = self._bind.setdefault(wlang, {})
                    ast = declast.check_attrs(value, bind, statements.BindArg)
                value = langdict.get("attrs")
                if value:
                    for argname, attrs in value.items():
                        if argname == "__line__":
                            continue
                        bind = statements.fetch_name_bind(
                            self._bind, wlang, argname)
                        bind.meta.update(attrs)

        # XXX - waring about unused fields in attrs
        error.cursor.pop_node(self)

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

        fmt_func = self.fmtdict

        fmt_func.update(dict(
            function_name=self.name,
            C_name_api = self.apply_C_API_option(self.name),
            F_name_api=self.apply_F_API_option(self.name),
        ))

        if fmtdict:
            self.fmtdict.update(fmtdict, replace=True)

    def clone(self):
        """Create a copy of a FunctionNode to use with C++ template
        or changing result to argument.
        """
        # Shallow copy everything.
        new = copy.copy(self)

        # new Scope with same inlocal and parent.
        new.fmtdict = self.fmtdict.clone()
        new.options = self.options.clone()
        new.wrap = WrapFlags(self.options)

        # Deep copy dictionaries to allow them to be modified independently.
        new.ast = copy.deepcopy(self.ast)
        new._fmtlang = copy.deepcopy(self._fmtlang)
        new._bind = {}
        new._generated_path = copy.deepcopy(self._generated_path)
        if new._orig_node is None:
            new._orig_node = self

        return new

    def may_have_args(self):
        # only FunctionNode may have args
        return True

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
          splicer:
            f: |
              blah blah blah

    _fmtmembers = {
      'RED': Scope(_fmt_func)

    }
    """

    def __init__(
            self, decl, parent, ast=None,
            format={}, options=None, splicer={}, **kwargs
    ):

        # From arguments
        self.parent = parent
        self.symtab = parent.symtab
        self.cxx_header = []
        self.linenumber = kwargs.get("__line__", "?")
        self.splicer = splicer

        self.options = util.Scope(parent.options)
        if options:
            self.options.update(options, replace=True)
        self.wrap = WrapFlags(self.options)

        #        self.default_format(parent, format, kwargs)
        self.user_fmt = format
        self.fmtdict = util.Scope(parent=parent.fmtdict)
        error.cursor.push_node(self)

        if not decl:
            raise RuntimeError("EnumNode missing decl")

        self.decl = decl
        if ast is None:
            ast = declast.check_decl(decl, self.symtab)
        if not isinstance(ast, declast.Declaration):
            raise RuntimeError("Declaration is not an enumeration: " + decl)
        enum_specifier = ast.enum_specifier
        if not isinstance(enum_specifier, declast.Enum):
            raise RuntimeError("Declaration is not an enumeration: " + decl)
        self.ast = enum_specifier
        self.name = enum_specifier.name

        # format for enum
        fmt_enum = self.fmtdict
        fmt_enum.enum_name = self.name
        if fmt_enum.cxx_class:
            fmt_enum.namespace_scope = (
                fmt_enum.namespace_scope + fmt_enum.cxx_class + "::"
            )
        fmt_enum.C_name_api = self.apply_C_API_option(self.name)
        fmt_enum.F_name_api = self.apply_F_API_option(self.name)
        self.eval_template("F_name_enum")
        self.eval_template("C_enum_type")

        # Format for each enum member.
        # Compute all names first since any expression must be converted to 
        # C or Fortran names.
        options = self.options
        fmtmembers = {}
        if enum_specifier.scope is not None:
            # members of 'class enum' must be qualified, add to scope.
            C_name_scope = self.parent.fmtdict.C_name_scope + self.name + "_"
            F_name_scope = self.parent.fmtdict.F_name_scope + self.name.lower() + "_"
        for member in enum_specifier.members:
            fmt = util.Scope(parent=fmt_enum)
            fmt.enum_member_name = member.name
            fmt.C_name_api = self.apply_C_API_option(member.name)
            fmt.F_name_api = self.apply_F_API_option(member.name)
            if enum_specifier.scope is not None:
                fmt.C_name_scope = C_name_scope
                fmt.F_name_scope = F_name_scope
            fmt.C_enum_member = wformat(options.C_enum_member_template, fmt)
            fmt.F_enum_member = wformat(options.F_enum_member_template, fmt)
            fmtmembers[member.name] = fmt

        # Compute enum values.
        # Required for Fortran since it will not implicitly generate values.
        # Expressions are assumed to be built up from other enum values.
        cvalue = 0
        fvalue = 0
        value_is_int = True
        for member in enum_specifier.members:
            fmt = fmtmembers[member.name]
            # evaluate value
            if member.value is not None:
                try:
                    cvalue = int(todict.print_node(member.value))
                    fvalue = cvalue
                    value_is_int = True
                except ValueError:
                    cvalue = todict.print_node_identifier(
                        member.value, fmtmembers, "C_enum_member")
                    fvalue = todict.print_node_identifier(
                        member.value, fmtmembers, "F_enum_member")
                    cbase = cvalue
                    fbase = fvalue
                    incr = 0
                    value_is_int = False
                fmt.C_value = cvalue # Only set if explicitly set by user.
            fmt.F_value = fvalue     # Always set.

            # Prepare for next value.
            if value_is_int:
                cvalue = cvalue + 1
                fvalue = cvalue
            else:
                incr += 1
                cvalue = "{}+{}".format(cbase, incr)
                fvalue = "{}+{}".format(fbase, incr)

        self._fmtmembers = fmtmembers
        # Headers required by template arguments.
        self.gen_headers_typedef = {}

        # Add to namespace
        self.scope = self.parent.scope + self.name + "::"
        ftypemap = self.symtab.lookup_typemap(options.F_enum_type)
        if ftypemap is None:
            error.cursor.ast(self.linenumber,
                             "Unknown type in F_enum_type: %s" % options.F_enum_type)
            ftypemap = self.symtab.lookup_typemap("int")  # recover from error
        fmt_enum.F_enum_kind = ftypemap.f_kind
        self.typemap = ast.typemap
        typemap.fill_enum_typemap(self, ftypemap)
        # also 'enum class foo' will alter scope
        error.cursor.pop_node(self)

######################################################################


class TypedefNode(AstNode):
    """
    Typedef.
    Includes builtin typedefs and from declarations.

    type name must be in a typemap.

    - decl: typedef int IndexType
    - fields:
        
    - splicer:
      c: []
      f: []
      py: [] ???
    """

    def __init__(self, name, parent, ast, fields,
                 format={}, options={}, splicer={},
                 **kwargs):
        """
        Args:
            name - 
            parent - 
            ast - declast.Declaration, typedef statement.
        """

        # From arguments
        self.name = name
        self.parent = parent
        self.symtab = parent.symtab
        self.cxx_header = []
        self.linenumber = kwargs.get("__line__", "?")
        self.splicer = splicer

        self.options = util.Scope(parent=parent.options)
        if options:
            self.options.update(options, replace=True)
        self.wrap = WrapFlags(self.options)

        self.fmtdict = util.Scope(parent.fmtdict)
        self.user_fmt = format
        self.default_format(parent, format)
        if self.user_fmt:
            self.fmtdict.update(self.user_fmt, replace=True)
        self.update_names()

        self.ast = ast
        self.user_fields = fields
        self._bind = {}                   # Access with get_arg_bind

        # save info from original type used in generated declarations.
        ntypemap = ast.typemap
        self.f_kind = ntypemap.f_kind
        self.f_module = ntypemap.f_module
        self.typemap = ntypemap
            
        typemap.fill_typedef_typemap(self)
        if fields:
            ntypemap.update(fields)

    def get_typename(self):
        return self.typemap.name

    def default_format(self, parent, format):
        """Set format dictionary."""
        self.fmtdict = util.Scope(
            parent=parent.fmtdict,
            cxx_type=self.name,
            typedef_name=self.name,
            C_name_api = self.apply_C_API_option(self.name),
            F_name_api = self.apply_F_API_option(self.name),
        )

    def update_names(self):
        """Update C and Fortran names.

        Called when reading declaration and
        when instantiation a class in GenFunctions.template_typedef.
        """
        fmt = self.fmtdict
        if self.wrap.c:
            if "C_name_typedef" not in self.user_fmt:
                self.reeval_template("C_name_typedef")
        else:
            # language=c, use original name.
            fmt.C_name_typedef = fmt.typedef_name
        if self.wrap.fortran:
            if "F_name_typedef" not in self.user_fmt:
                self.reeval_template("F_name_typedef")

    def clone(self):
        """Create a copy of a TypedefNode to use with C++ template.
        """
        # Shallow copy everything.
        new = copy.copy(self)

        # new Scope with same inlocal and parent.
        new.fmtdict = self.fmtdict.clone()
        new.options = self.options.clone()
        new.wrap = WrapFlags(self.options)
        return new

    def clone_post_class(self, targs):
        """Steps to clone typedef after class has been instantiated.

        Need to create a new typemap for typedefs within a templated class.
        """
        self.update_names()
        type_name = util.wformat("{namespace_scope}{class_scope}{cxx_type}", self.fmtdict)

        ntypemap = self.typemap.clone_as(type_name)
        self.typemap = ntypemap
        self.parent.symtab.register_typemap(type_name, ntypemap)
        ntypemap.is_typedef = True
        typemap.fill_typedef_typemap(self)


######################################################################


class VariableNode(AstNode):
    """
        - decl: int var
          options:
             bar: 4
          format:
             baz: 4

    _bind = {
       'f': {   statements.BindArg
          stmt: Scope
          meta: collections.defaultdict(lambda: None)
          fmtdict:  Scope(_fmtfunc)
        }
    }

    Args:
        ast - If None, compute from decl.
    """

    def __init__(
        self, decl, parent, format={}, ast=None, options=None, **kwargs
    ):

        # From arguments
        self.parent = parent
        self.symtab = parent.symtab
        self.linenumber = kwargs.get("__line__", "?")

        self.options = util.Scope(parent=parent.options)
        if options:
            self.options.update(options, replace=True)
        self.wrap = WrapFlags(self.options)

        #        self.default_format(parent, format, kwargs)
        self.user_fmt = format
        self.fmtdict = util.Scope(parent=parent.fmtdict)
        self._bind = {}

        if not decl:
            raise RuntimeError("VariableNode missing decl")

        self.decl = decl
        if ast is None:
            ast = declast.check_decl(decl, self.symtab)
        if not isinstance(ast, declast.Declaration):
            # GGG - only declarations in stucts?
            raise RuntimeError("Declaration is not a structure: " + decl)
        if ast.declarator.params is not None:
            # 'void foo()' instead of 'void foo'
            raise RuntimeError("Arguments given to variable:", ast.gen_decl())
        self.ast = ast
        self.name = ast.declarator.user_name

        # format for struct
        fmt_var = self.fmtdict

        # Treat similar to class
        #        fmt_struct.class_scope = self.name + '::'
        fmt_var.field_name = ast.declarator.name
        fmt_var.variable_name = self.name
        fmt_var.variable_lower = fmt_var.variable_name.lower()
        fmt_var.variable_upper = fmt_var.variable_name.upper()

        ntypemap = ast.typemap
        fmt_var.c_type = ntypemap.c_type
        fmt_var.cxx_type = ntypemap.cxx_type

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

    def parse_instantiation(self, symtab):
        """Parse instantiation (ex. <int>) and set list of Declarations."""
        parser = declast.Parser(self.instantiation, symtab)
        self.asts = parser.template_argument_list()


######################################################################

class FortranGeneric(object):
    """Information used to create a fortran generic version of a function.

    generic : str
        "(double arg)"
    decls : [ declast.Declaration ]
        A parse list of generic argument.
    """
    def __init__(self, generic, fmtdict=None, options=None,
                 function_suffix="X", linenumber="?",
                 decls=None):
        self.generic = generic
        self.fmtdict = fmtdict
        self.options = options
        self.function_suffix = function_suffix
        self.linenumber = linenumber
        self.decls = decls
        self.first = 0      # First decl which is generic, 0=no arguments.

    def parse_generic(self, symtab):
        """Parse argument list (ex. int arg1, float *arg2)
        and set list of Declarations."""
        parser = declast.Parser(self.generic, symtab)
        self.decls = parser.parameter_list()

    def __repr__(self):
        return "<FortranGeneric({}>".format(self.generic)


def trim_fortran_generic_decls(fortran_generic, nargs):
    """When generating a function with default arguments,
    return a new list of FortranGeneric with the decls trimmed
    to match the number of arguments.
    """
    newlst = []
    for generic in fortran_generic:
        if nargs < generic.first:
            continue
        new = copy.copy(generic)
        new.decls = generic.decls[:nargs]
        newlst.append(new)
    return newlst
    
######################################################################

class PromoteWrap(visitor.Visitor):
    """Promote wrap_x options up to container.

    For example:
       options:
         wrap_f: False
       declarations:
       - decl: void Func(void)
         options:
           wrap_f: True

    Since the function's wrap_f option is True, the library should set
    library.wrap.fortran to True as well.  Likewise for other
    containers: namespace, class.

    """
    def visit_LibraryNode(self, node):
        wrap = node.wrap
        for cls in node.classes:
            self.visit(cls)
            wrap.accumulate(cls.wrap)
        for en in node.enums:
            wrap.accumulate(en.wrap)
        for fcn in node.functions:
            wrap.accumulate(fcn.wrap)
        for ns in node.namespaces:
            self.visit(ns)
            wrap.accumulate(ns.wrap)
        for typ in node.typedefs:
            wrap.accumulate(typ.wrap)
        for var in node.variables:
            wrap.accumulate(var.wrap)

    def visit_ClassNode(self, node):
        wrap = node.wrap
        for cls in node.classes:
            self.visit(cls)
            wrap.accumulate(cls.wrap)
        for en in node.enums:
            wrap.accumulate(en.wrap)
        for fcn in node.functions:
            wrap.accumulate(fcn.wrap)
        for typ in node.typedefs:
            wrap.accumulate(typ.wrap)
        for var in node.variables:
            wrap.accumulate(var.wrap)

    def visit_NamespaceNode(self, node):
        wrap = node.wrap
        for cls in node.classes:
            self.visit(cls)
            wrap.accumulate(cls.wrap)
        for en in node.enums:
            wrap.accumulate(en.wrap)
        for fcn in node.functions:
            wrap.accumulate(fcn.wrap)
        for ns in node.namespaces:
            self.visit(ns)
            wrap.accumulate(ns.wrap)
        for typ in node.typedefs:
            wrap.accumulate(typ.wrap)
        for var in node.variables:
            wrap.accumulate(var.wrap)


def promote_wrap(node):
    """Promote wrap options to parent containers.

    Parameters
    ----------
    node : LibraryNode
        Could be any AstNode subclass
    """
    visitor = PromoteWrap()
    return visitor.visit(node)

######################################################################
# Parse yaml file
######################################################################


def clean_dictionary(ddct):
    """YAML converts some blank fields to None,
    but we want blank.
    return True if no errors detected.
    """
    ok = True
    for key in ["cxx_header", "namespace"]:
        if key in ddct and ddct[key] is None:
            ddct[key] = ""

    linenumber=ddct.get("__line__", "?")
    if "default_arg_suffix" in ddct:
        default_arg_suffix = ddct["default_arg_suffix"]
        if not isinstance(default_arg_suffix, list):
            error.cursor.ast(linenumber, "field 'default_arg_suffix' must be a list")
            ok = False
        for i, value in enumerate(ddct["default_arg_suffix"]):
            if value is None:
                ddct["default_arg_suffix"][i] = ""

    if "format" in ddct:
        fmtdict = ddct["format"]
        for key in ["function_suffix"]:
            if key in fmtdict and fmtdict[key] is None:
                fmtdict[key] = ""

    #  cxx_template:
    #  - instantiation: <int, long>
    #  - instantiation: <float, double>
    if "cxx_template" in ddct:
        # Convert to list of TemplateArgument instances
        cxx_template = ddct["cxx_template"]
        if not isinstance(cxx_template, list):
            error.cursor.ast(linenumber, "field 'cxx_template' must be a list")
            ok = False
        else:
            newlst = []
            for dct in cxx_template:
                if not isinstance(dct, dict):
                    error.cursor.ast(linenumber, "field 'cxx_template' must be a list of dictionaries")
                    ok = False
                elif "instantiation" not in dct:
                    linenumber = dct.get("__line__", "?")
                    error.cursor.ast(linenumber, 
                        "'instantation' must be defined for each dictionary in 'cxx_template'"
                    )
                    ok = False
                else:
                    newlst.append(
                        TemplateArgument(
                            dct["instantiation"],
                            fmtdict=dct.get("format", None),
                            options=dct.get("options", None),
                        )
                    )
            ddct["cxx_template"] = newlst

    #  fortran_generic:
    #  - decl: float arg
    #  - decl: double arg
    if "fortran_generic" in ddct:
        # Convert to list of TemplateArgument instances
        fortran_generic = ddct["fortran_generic"]
        if not isinstance(fortran_generic, list):
            if isinstance(fortran_generic, dict):
                linenumber=fortran_generic.get("__line__", "?")
            else:
                linenumber=ddct.get("__line__", "?")
            error.cursor.ast(linenumber, "field 'fortran_generic' must be a list")
            ok = False
        else:
            newlst = []
            isuffix = 0
            for dct in fortran_generic:
                if not isinstance(dct, dict):
                    linenumber=ddct.get("__line__", "?")
                    error.cursor.ast(
                        linenumber,
                        "field 'fortran_generic' must be a list of dictionaries")
                    ok = False
                elif "decl" not in dct:
                    linenumber=dct.get("__line__", "?")
                    error.cursor.ast(
                        linenumber,
                        "'decl' must be defined for each dictionary in 'fortran_generic'"
                    )
                else:
                    newlst.append(
                        FortranGeneric(
                            dct["decl"],
                            fmtdict=dct.get("format", None),
                            options=dct.get("options", None),
                            function_suffix=dct.get("function_suffix", "_" + str(isuffix)),
                            linenumber=linenumber,
                        )
                    )
                    isuffix += 1
            ddct["fortran_generic"] = newlst

    if "fields" in ddct:
        fields = ddct["fields"]
        if not isinstance(fields, dict):
            error.cursor.ast(linenumber, "'fields' must be a dictionary")
            ok = False
    return ok


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


def listify_cleanup(value):
    """Clean up top level splicer_code

    Convert newline delimited strings into a list of strings.
    Remove trailing newline which will generate a blank line.
    Or replace None with "" in a list.

    Used with nested dictionaries used to mirror scopes.

     CXX_definitions: |
       // Add some text from splicer
       // And another line
     namespace:
       ns0:
         CXX_definitions:
         - // lines from explict splicer - namespace ns0

    """
    
    if isinstance(value, dict):
       new = {}
       for key, item in value.items():
           new[key] = listify_cleanup(item)
    elif isinstance(value, str):
        new = value.split("\n")
        if value[-1] == "\n":
            new.pop()
    elif isinstance(value, list):
        new = ["" if v is None else v for v in value]
    else:
        new = [ str(value) ]
    return new

def listify(entry, names):
    """
    Convert newline delimited strings into a list of strings.
    Remove trailing newline which will generate a blank line.
    Or replace None with "" in a list.
    Accept a dictionary and return a new dictionary.

    splicer:
      c: |
        // line 1
        // line 2
      c_buf:
        - // Test adding a blank line below.
        -

    fstatements:
      f:
        local_var: pointer
        call: |
           blah blah
           yada yada yada

    Args:
        entry - dictionary
        names - key names which must be lists.
    """
    new = {}
    for key, value in entry.items():
        if isinstance(value, dict):
            new[key] = listify(value, names)
        elif key in names:
            if isinstance(value, str):
#              if value[-1] == "\n":
#                  new[key] = [ value[:-1] ]
#              else:
#                  new[key] = [ value ]
                new[key] = value.split("\n")
                if value[-1] == "\n":
                    new[key].pop()
            elif isinstance(value, list):
                new[key] = ["" if v is None else v for v in value]
            else:
                new[key] = [ str(value) ]
        else:
            new[key] = value
    return new


def add_declarations(parent, node, symtab):
    """Add "declarations" from node dictionary.

    node is from a YAML file.
    """
    if "declarations" not in node:
        return
    if not node["declarations"]:
        return

    for subnode in node["declarations"]:
        if "block" in subnode:
            dct = copy.copy(subnode)
            if clean_dictionary(dct):
                blk = BlockNode(parent, **dct)
                add_declarations(blk, subnode, symtab)
        elif "decl" in subnode:
            # copy before clean to avoid changing input dict
            dct = copy.copy(subnode)
            if not clean_dictionary(dct):
                continue
            decl = dct["decl"]
            del dct["decl"]

            if "fstatements" in dct:
                dct["fstatements"] = listify(dct["fstatements"], [
                    "c_pre_call", "c_post_call", "c_final",
                    "c_return",

                    "pre_call", "call", "post_call", "final", "ret",
                    "declare",
                    "post_parse",
                    "declare_capsule", "post_call_capsule", "fail_capsule",
                    "declare_keep", "post_call_keep", "fail_keep",
                    "cleanup", "fail",
                ])
            if "splicer" in dct:
                dct["splicer"] = listify(
                    dct["splicer"],["c", "c_buf", "f", "py"]
                )

            old = symtab.save_depth()
            try:
                declnode = parent.add_declaration(decl, **dct)
            except error.ShroudParseError as err:
                linenumber = dct.get("__line__", "?")
                error.get_cursor().ast(linenumber, decl, err)
            except error.ShroudError as err:
                linenumber = dct.get("__line__", "?")
#                error.get_cursor().warning(err.message)
                error.get_cursor().generate(err.message)
            else:
                add_declarations(declnode, subnode, symtab)
            symtab.restore_depth(old)

        else:
            linenumber = subnode.get("__line__", "?")
            error.get_cursor().ast(linenumber, "Expected 'block' or 'decl' in group")


def create_library_from_dictionary(node, symtab):
    """Create a library and add classes and functions from node.
    Typically, node is a dictionary defined via YAML.

    library: name
    classes:
    - name: Class1
    functions:
    - decl: void func1()

    Do some checking on the input.
    Every class must have a name.
    """

    cursor = error.get_cursor()
    cursor.nwarning = 0  # reset for tests
    cursor.push_phase("Create library")
    if "copyright" in node:
        clean_list(node["copyright"])

    clean_dictionary(node)
    library = LibraryNode(symtab=symtab, **node)
####
    if "typemap" in node:
        symtab.stash_stack()
        # Add typemaps to SymbolTable.
        # list of dictionaries
        typemaps = symtab.typemaps
        for subnode in node["typemap"]:
            # Update fields for a type. For example, set cpp_if
            if "type" not in subnode:
                linenumber = subnode.get("__line__", "?")
                error.cursor.ast(linenumber, "typemap must have 'type' member")
                continue
            if "fields" not in subnode:
                linenumber = subnode.get("__line__", "?")
                error.cursor.ast(linenumber, "typemap must have 'fields' member")
                continue
            key = subnode["type"]  # XXX make sure fields exist
            fields = subnode.get("fields")
            ntypemap = typemaps.get(key, None)
            if ntypemap:
                if fields:
                    ntypemap.update(fields)
            else:
                # Create new typemap
                base = fields.get("base", "")
                if base == "shadow":
                    typemap.create_class_typemap_from_fields(
                        key, fields, library
                    )
                elif base == "struct":
                    typemap.create_struct_typemap_from_fields(
                        key, fields, library
                    )
                elif base in ["integer", "real", "complex"]:
                    ntypemap = typemap.create_native_typemap_from_fields(
                        key, fields, library
                    )
                    ntypemap.export = True
                else:
                    linenumber = fields.get("__line__", "?")
                    error.cursor.ast(linenumber, "Unknown value for 'base' in 'typemap'")
        symtab.restore_stack()

    add_declarations(library.wrap_namespace, node, library.symtab)

    if "splicer_code" in node: 
        new = {}
        for key, value in node["splicer_code"].items():
            if key in ["c", "f", "py"]:
                new[key] = listify_cleanup(value)
        node["splicer_code"] = new
    cursor.pop_phase("Create library")
    cursor.check_for_warnings()
    
    return library
