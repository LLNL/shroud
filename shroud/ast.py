# Copyright (c) 2017-2018, Lawrence Livermore National Security, LLC.
# Produced at the Lawrence Livermore National Laboratory
# 
# LLNL-CODE-738041.
# All rights reserved.
#  
# This file is part of Shroud.  For details, see
# https://github.com/LLNL/shroud. Please also read shroud/LICENSE.
#  
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#  
# * Redistributions of source code must retain the above copyright
#   notice, this list of conditions and the disclaimer below.
# 
# * Redistributions in binary form must reproduce the above copyright
#   notice, this list of conditions and the disclaimer (as noted below)
#   in the documentation and/or other materials provided with the
#   distribution.
# 
# * Neither the name of the LLNS/LLNL nor the names of its contributors
#   may be used to endorse or promote products derived from this
#   software without specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL LAWRENCE
# LIVERMORE NATIONAL SECURITY, LLC, THE U.S. DEPARTMENT OF ENERGY OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# 
########################################################################
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
    def option_to_fmt(self, fmtdict):
        """Set fmt based on options dictionary.
        """
        for name in ['C_prefix', 'F_C_prefix',
                     'C_result', 'F_result', 'F_derived_member',
                     'PY_result', 'LUA_result',
                     'C_this', 'CXX_this', 'F_this',
                     'C_string_result_as_arg', 'F_string_result_as_arg',
                     'C_header_filename_suffix',
                     'C_impl_filename_suffix',
                     'F_filename_suffix',
                     'PY_prefix',
                     'PY_header_filename_suffix',
                     'PY_impl_filename_suffix',
                     'LUA_prefix',
                     'LUA_header_filename_suffix',
                     'LUA_impl_filename_suffix',
        ]:
            if self.options.inlocal(name):
                raise DeprecationWarning("Setting option {} for {}, change to format group".format(
                    name, self.__class__.__name__))

    def eval_template(self, name, tname='', fmt=None):
        """If a format has not been explicitly set, set from template."""
        if fmt is None:
            fmt = self.fmtdict
        if not fmt.inlocal(name):
            tname = name + tname + '_template'
            setattr(fmt, name, util.wformat(self.options[tname], fmt))

######################################################################

class NamespaceMixin(object):
    def add_class(self, name, **kwargs):
        """Add a class.
        """
        node = ClassNode(name, self, **kwargs)
        self.classes.append(node)
        self.symbols[name] = node
        return node

    def add_declaration(self, decl, **kwargs):
        """parse decl and add corresponding node.
        decl - declaration
        """
        # parse declaration to find out what it is
        if isinstance(self, ClassNode):
            cls_name = self.name
        else:
            cls_name = None
        ast = declast.check_decl(decl, namespace=self,
                                 template_types=kwargs.get('cxx_template', {}).keys())
        if isinstance(ast, declast.Declaration):
            self.add_function(decl, ast=ast, **kwargs)
        elif isinstance(ast, declast.Enum):
            self.add_enum(decl, ast=ast, **kwargs)
        else:
            raise RuntimeError("Error parsing " + decl)

    def add_enum(self, decl, parentoptions=None, ast=None, **kwargs):
        """Add an enumeration.
        """
        node = EnumNode(decl, parent=self, parentoptions=parentoptions,
                        ast=ast, **kwargs)
        self.enums.append(node)
        self.symbols[node.name] = node
        return node

    def add_function(self, decl, parentoptions=None, ast=None, **kwargs):
        """Add a function.

        decl - C/C++ declaration of function
        ast  - parsed declaration. None if not yet parsed.
        """
        fcnnode = FunctionNode(decl, parent=self, parentoptions=parentoptions,
                               ast=ast, **kwargs)
        self.functions.append(fcnnode)
        return fcnnode

    def add_namespace(self, name, parentoptions=None, **kwargs):
        """Add an namespace
        """
        node = NamespaceNode(name, parent=self, parentoptions=parentoptions,
                             **kwargs)
        self.symbols[name] = node
        return node

    def add_typedef(self, name):
        """Add a typedef.
        """
        node = TypedefNode(name, parent=self)
        self.symbols[name] = node
        return node

######################################################################

class LibraryNode(AstNode, NamespaceMixin):
    def __init__(self,
                 cxx_header='',
                 format=None,
                 language='c++',
                 library='default_library',
                 options=None,
                 **kwargs):
        """Create LibraryNode.

        fields = value
        options:
        classes:
        functions:

        """
        # From arguments
        self.cxx_header = cxx_header
        self.language = language.lower()
        if self.language not in ['c', 'c++']:
            raise RuntimeError("language must be 'c' or 'c++'")
        self.library = library

        self.classes = []
        self.enums = []
        self.functions = []
        # Each is given a _function_index when created.
        self.function_index = []
        self.options = self.default_options()
        if options:
            self.options.update(options, replace=True)

        self.F_module_dependencies = []     # unused

        self.copyright = kwargs.setdefault('copyright', [])
        self.patterns = kwargs.setdefault('patterns', [])

        self.default_format(format, kwargs)

        # namespace
        self.scope = ''
        self.symbols = {}
        self.using = []
        declast.global_namespace = self
        self.create_std_names()

##### namespace behavior

    def create_std_names(self):
        """Add standard types to the Library."""
        self.add_typedef('size_t')
        self.add_typedef('MPI_Comm')
        create_std_namespace(self)   # add 'std::' to library
        self.using_directive('std')

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

#####

    def default_options(self):
        """default options."""
        def_options = util.Scope(
            parent=None,
            debug=False,   # print additional debug info
            C_line_length=72,

            F_line_length=72,
            F_module_per_class=True,
            F_string_len_trim=True,
            F_force_wrapper=False,
            F_standard=2003,

            wrap_c=True,
            wrap_fortran=True,
            wrap_python=False,
            wrap_lua=False,

            doxygen=True,       # create doxygen comments
            show_splicer_comments=True,

            # blank for functions, set in classes.
            class_prefix_template='{class_lower}_',

            YAML_type_filename_template='{library_lower}_types.yaml',

            C_header_filename_library_template='wrap{library}.{C_header_filename_suffix}',
            C_impl_filename_library_template='wrap{library}.{C_impl_filename_suffix}',

            C_header_filename_class_template='wrap{cxx_class}.{C_header_filename_suffix}',
            C_impl_filename_class_template='wrap{cxx_class}.{C_impl_filename_suffix}',

            C_enum_template='{C_prefix}{class_prefix}{enum_name}',
            C_enum_member_template='{enum_member_name}',

            C_name_template=(
                '{C_prefix}{class_prefix}{underscore_name}{function_suffix}'),

            C_var_len_template = 'N{c_var}',         # argument for result of len(arg)
            C_var_trim_template = 'L{c_var}',        # argument for result of len_trim(arg)
            C_var_size_template = 'S{c_var}',        # argument for result of size(arg)

            # Fortran's names for C functions
            F_C_name_template=(
                '{F_C_prefix}{class_prefix}{underscore_name}{function_suffix}'),

            F_enum_member_template=(
                '{class_prefix}{enum_lower}_{enum_member_lower}'),

            F_name_impl_template=(
                '{class_prefix}{underscore_name}{function_suffix}'),

            F_name_function_template='{underscore_name}{function_suffix}',
            F_name_generic_template='{underscore_name}',

            F_module_name_library_template='{library_lower}_mod',
            F_impl_filename_library_template='wrapf{library_lower}.{F_filename_suffix}',

            F_module_name_class_template='{class_lower}_mod',
            F_impl_filename_class_template='wrapf{cxx_class}.{F_filename_suffix}',
            F_abstract_interface_subprogram_template='{underscore_name}_{argname}',
            F_abstract_interface_argument_template='arg{index}',

            LUA_module_name_template='{library_lower}',
            LUA_module_filename_template=(
                'lua{library}module.{LUA_impl_filename_suffix}'),
            LUA_header_filename_template=(
                'lua{library}module.{LUA_header_filename_suffix}'),
            LUA_userdata_type_template='{LUA_prefix}{cxx_class}_Type',
            LUA_userdata_member_template='self',
            LUA_module_reg_template='{LUA_prefix}{library}_Reg',
            LUA_class_reg_template='{LUA_prefix}{cxx_class}_Reg',
            LUA_metadata_template='{cxx_class}.metatable',
            LUA_ctor_name_template='{cxx_class}',
            LUA_name_template='{function_name}',
            LUA_name_impl_template='{LUA_prefix}{class_prefix}{underscore_name}',

            PY_module_filename_template=(
                'py{library}module.{PY_impl_filename_suffix}'),
            PY_header_filename_template=(
                'py{library}module.{PY_header_filename_suffix}'),
            PY_helper_filename_template=(
                'py{library}helper.{PY_impl_filename_suffix}'),
            PY_PyTypeObject_template='{PY_prefix}{cxx_class}_Type',
            PY_PyObject_template='{PY_prefix}{cxx_class}',
            PY_type_filename_template=(
                'py{cxx_class}type.{PY_impl_filename_suffix}'),
            PY_name_impl_template=(
                '{PY_prefix}{class_prefix}{function_name}{function_suffix}'),
            # names for type methods (tp_init)
            PY_type_impl_template=(
                '{PY_prefix}{cxx_class}_{PY_type_method}{function_suffix}'),
            )
        return def_options

    def default_format(self, format, kwargs):
        """Set format dictionary.

        Values based off of library variables and
        format templates in options.
        """

        fmt_library = util.Scope(
            parent=None,

            C_bufferify_suffix='_bufferify',
            C_prefix = self.library.upper()[:3] + '_',  # function prefix
            C_result = 'rv',        # return value
            C_argument = 'SH_',
            c_temp = 'SHT_',
            C_local = 'SHC_',
            C_this = 'self',

            C_custom_return_type = '',  # assume no value

            CXX_this = 'SH_this',
            CXX_local = 'SHCXX_',
            cxx_class='',     # Assume no class
            class_scope='',

            F_C_prefix='c_',
            F_derived_member = 'voidptr',
            F_name_associated = 'associated',
            F_name_instance_get = 'get_instance',
            F_name_instance_set = 'set_instance',
            F_result = 'SHT_rv',
            F_this = 'obj',

            C_string_result_as_arg = 'SHF_rv',
            F_string_result_as_arg = '',

            PY_result = 'SHTPy_rv',      # Create PyObject for result
            LUA_result = 'rv',

            LUA_prefix = 'l_',
            LUA_state_var = 'L',
            LUA_this_call = '',

            PY_prefix = 'PY_',
            PY_module_name = self.library.lower(),
            PY_this_call = '',

            library = self.library,
            library_lower = self.library.lower(),
            library_upper = self.library.upper(),

        # set default values for fields which may be unset.
            class_prefix = '',   # expand to blanks for library
#           c_ptr = '',
#           c_const = '',
            CXX_this_call = '',
            CXX_template = '',
            C_pre_call = '',
            C_post_call = '',
            function_suffix = '',   # assume no suffix
            namespace_scope = '',
        )

        fmt_library.F_filename_suffix = 'f'

        if self.language == 'c':
            fmt_library.C_header_filename_suffix = 'h'
            fmt_library.C_impl_filename_suffix = 'c'

            fmt_library.LUA_header_filename_suffix = 'h'
            fmt_library.LUA_impl_filename_suffix = 'c'

            fmt_library.stdlib  = ''
        else:
            fmt_library.C_header_filename_suffix = 'h'
            fmt_library.C_impl_filename_suffix = 'cpp'

            fmt_library.LUA_header_filename_suffix = 'hpp'
            fmt_library.LUA_impl_filename_suffix = 'cpp'

            fmt_library.stdlib  = 'std::'

        for n in ['C_header_filename', 'C_impl_filename',
                  'F_module_name', 'F_impl_filename',
                  'LUA_module_name', 'LUA_module_reg', 'LUA_module_filename', 'LUA_header_filename',
                  'PY_module_filename', 'PY_header_filename', 'PY_helper_filename',
                  'YAML_type_filename'
        ]:
            if n in kwargs:
                raise DeprecationWarning("Setting field {} in library, change to format group".format(
                    n))

        self.option_to_fmt(fmt_library)

        if format:
            fmt_library.update(format, replace=True)

        self.fmtdict = fmt_library

        # default some format strings based on other format strings
        self.eval_template('C_header_filename', '_library')
        self.eval_template('C_impl_filename', '_library')
        # All class/methods and functions may go into this file or
        # just functions.
        self.eval_template('F_module_name', '_library')
        self.eval_template('F_impl_filename', '_library')

######################################################################

class BlockNode(AstNode, NamespaceMixin):
    """Create a Node to simulate a curly block.
    A block can contain options, format, and declarations.
    The declarations within a BlockNode inherit options of the block.
    This makes it easier to change options for a group of functions.
    Declarations are added to parent.

    Blocks can be added to a LibraryNode, NamespaceNode or ClassNode.
    """
    def __init__(self, parent,
                 format=None,
                 options=None,
                 **kwargs):
        # From arguments
        self.parent = parent

        self.enums = parent.enums
        self.functions = parent.functions
        self.classes = parent.classes

        self.options = util.Scope(parent=parent.options)
        if options:
            self.options.update(options, replace=True)

        self.fmtdict = util.Scope(parent=parent.fmtdict)
        if format:
            self.fmtdict.update(format, replace=True)

    def unqualified_lookup(self, name):
        """Look for symbols within library (global namespace). """
        return self.parent.unqualified_lookup(name)

######################################################################

class NamespaceNode(AstNode, NamespaceMixin):
    def __init__(self, name, parent,
                 format=None,
                 options=None,
                 **kwargs):
        """Create NamespaceNode.

        parent may be LibraryNode or NamespaceNode.
        """
        # From arguments
        self.name = name
        self.parent = parent

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
            break

        self.options = util.Scope(parent=parent.options)
        if options:
            self.options.update(options, replace=True)

        self.default_format(parent, format, kwargs)

        # add to symbol table
        self.scope = self.parent.scope + self.name + '::'
        self.symbols = {}
        self.using = []

##### namespace behavior

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

    def default_format(self, parent, format, kwargs):
        """Set format dictionary."""

        self.fmtdict = util.Scope(
            parent = parent.fmtdict,
        )

        fmt_ns = self.fmtdict
        fmt_ns.namespace_scope = parent.fmtdict.namespace_scope + self.name + '::\t'
        fmt_ns.CXX_this_call = fmt_ns.namespace_scope
        fmt_ns.LUA_this_call = fmt_ns.namespace_scope
        fmt_ns.PY_this_call = fmt_ns.namespace_scope

        if format:
            fmt_ns.update(format, replace=True)

######################################################################

class ClassNode(AstNode, NamespaceMixin):
    is_class = True
    def __init__(self, name, parent,
                 cxx_header='',
                 format=None,
                 options=None,
                 **kwargs):
        """Create ClassNode.
        """
        # From arguments
        self.name = name
        self.parent = parent
        self.cxx_header = cxx_header

        self.enums = []
        self.functions = []

        self.python = kwargs.get('python', {})
        self.cpp_if = kwargs.get('cpp_if', None)

        self.options = util.Scope(parent=parent.options)
        if options:
            self.options.update(options, replace=True)

        self.default_format(parent, format, kwargs)

        # add to namespace
        self.typename = self.parent.scope + self.name
        self.scope = self.typename + '::'
        self.symbols = {}
        self.typedef = typemap.create_class_typedef(self)
        self.typedef_name = self.typedef.name   # fully qualified name

##### namespace behavior

    def qualified_lookup(self, name):
        """Look for symbols within class.
        -- Only enums
        """
        return self.symbols.get(name, None)

    def unqualified_lookup(self, name):
        """Look for name in class or its parent."""
        if name in self.symbols:
            return self.symbols[name]
        return self.parent.unqualified_lookup(name)

#####

    def default_format(self, parent, format, kwargs):
        """Set format dictionary."""

        for n in ['C_header_filename', 'C_impl_filename',
                  'F_derived_name', 'F_impl_filename', 'F_module_name',
                  'LUA_userdata_type', 'LUA_userdata_member', 'LUA_class_reg',
                  'LUA_metadata', 'LUA_ctor_name',
                  'PY_PyTypeObject', 'PY_PyObject', 'PY_type_filename',
                  'class_prefix'
        ]:
            if n in kwargs:
                raise DeprecationWarning("Setting field {} in class {}, change to format group".format(
                    n, self.name))

        self.fmtdict = util.Scope(
            parent = parent.fmtdict,

            class_scope = self.name + '::',
            cxx_class = self.name,
            class_lower = self.name.lower(),
            class_upper = self.name.upper(),

            F_derived_name = self.name.lower(),
        )

        fmt_class = self.fmtdict
        if format:
            fmt_class.update(format, replace=True)

        self.eval_template('class_prefix')

        # Only one file per class for C.
        self.eval_template('C_header_filename', '_class')
        self.eval_template('C_impl_filename', '_class')

        if self.options.F_module_per_class:
            self.eval_template('F_module_name', '_class')
            self.eval_template('F_impl_filename', '_class')

    def add_namespace(self, **kwargs):
        """Replace method inherited from NamespaceMixin."""
        raise RuntimeError("Cannot add a namespace to a class")

######################################################################

class FunctionNode(AstNode):
    """

    - decl:
      cxx_template:
        ArgType:
        - int
        - double


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

    _function_index  - sequence number function,
                       used in lieu of a pointer
    _generated       - who generated this function
    _PTR_F_C_index   - Used by fortran wrapper to find index of
                       C function to call
    _PTR_C_CXX_index - Used by C wrapper to find index of C++ function
                       to call

    """
    def __init__(self, decl, parent,
                 format=None,
                 parentoptions=None,
                 ast=None,
                 options=None,
                 **kwargs):
        self.options = util.Scope(parent=parentoptions or parent.options)
        if options:
            self.options.update(options, replace=True)

        self.default_format(parent, format, kwargs)

        # working variables
        self._CXX_return_templated = False
        self._PTR_C_CXX_index = None
        self._PTR_F_C_index = None
        self._cxx_overload = None
        self.declgen = None              #  generated declaration.
        self._default_funcs = []         #  generated default value functions  (unused?)
        self._function_index = None
        self._fmtargs = {}
        self._fmtresult = {}
        self._function_index = None
        self._generated = False
        self._has_default_arg = False
        self._nargs = None
        self._overloaded = False

#        self.function_index = []

        self.default_arg_suffix = kwargs.get('default_arg_suffix', [])
        self.cpp_if = kwargs.get('cpp_if', None)
        self.cxx_template = kwargs.get('cxx_template', {})
        self.doxygen = kwargs.get('doxygen', {})
        self.fortran_generic = kwargs.get('fortran_generic', {})
        self.return_this = kwargs.get('return_this', False)

        if not decl:
            raise RuntimeError("FunctionNode missing decl")

        self.decl = decl
        if ast is None:
            # parse decl and add to dictionary
            if isinstance(parent, ClassNode):
                cls_name = parent.name
            else:
                cls_name = None
            template_types = self.cxx_template.keys()

            ast = declast.check_decl(decl,
                                     namespace=parent,
                                     template_types=template_types)
        self.ast = ast

        # add any attributes from YAML files to the ast
        if 'attrs' in kwargs:
            attrs = kwargs['attrs']
            for arg in ast.params:
                name = arg.name
                if name in attrs:
                    arg.attrs.update(attrs[name])
        if 'fattrs' in kwargs:
            ast.fattrs.update(kwargs['fattrs'])
        # XXX - waring about unused fields in attrs
                                    
        if ast.params is None:
            # 'void foo' instead of 'void foo()'
            raise RuntimeError("Missing arguments to function:", ast.gen_decl())

        fmt_func = self.fmtdict
        fmt_func.function_name = ast.name
        fmt_func.underscore_name = util.un_camel(fmt_func.function_name)

    def default_format(self, parent, format, kwargs):

        # Move fields from kwargs into instance
        for n in [
                'C_code',
#               'C_error_pattern',
                 'C_name',
                'C_post_call', 'C_post_call_buf',
                'C_return_code', 'C_return_type',
                'F_C_name',
                'F_code',
                'F_name_function', 'F_name_generic', 'F_name_impl',
                'LUA_name', 'LUA_name_impl',
#                'PY_error_pattern',
                'PY_name_impl',
                'function_suffix'
        ]:
            if n in kwargs:
                raise DeprecationWarning("Setting field {} in function, change to format group".format(
                    n))

        # Move fields from kwargs into instance
        for n in [
                'C_error_pattern', 'PY_error_pattern',
        ]:
            setattr(self, n, kwargs.get(n, None))

        self.fmtdict = util.Scope(parent.fmtdict)

        self.option_to_fmt(self.fmtdict)
        if format:
            self.fmtdict.update(format, replace=True)
            if 'C_return_type' in format:
                # wrapc.py will overwrite C_return_type.
                # keep original value for wrapf.py.
                self.fmtdict.C_custom_return_type = format['C_return_type']

    def clone(self):
        """Create a copy of a function node to use with C++ template
        or changing result to argument.
        """
        # Shallow copy everything
        new = copy.copy(self)

        # new Scope with same inlocal and parent
        new.fmtdict = self.fmtdict.clone()
        new.options = self.options.clone()
    
        # deep copy dictionaries
        new.ast = copy.deepcopy(self.ast)
        new._fmtargs = copy.deepcopy(self._fmtargs)
        new._fmtresult = copy.deepcopy(self._fmtresult)
    
        return new

######################################################################

class EnumNode(AstNode):
    """
        enums:
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
    def __init__(self, decl, parent,
                 format=None,
                 parentoptions=None,
                 ast=None,
                 options=None,
                 **kwargs):

        # From arguments
        self.parent = parent

        self.options = util.Scope(parent=parentoptions or parent.options)
        if options:
            self.options.update(options, replace=True)

#        self.default_format(parent, format, kwargs)
        self.fmtdict = util.Scope(
            parent = parent.fmtdict,
        )

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
            fmt_enum.namespace_scope =  fmt_enum.namespace_scope + fmt_enum.cxx_class + '::'

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
        self.typename = self.parent.scope + self.name
        self.scope = self.typename + '::'
        self.typedef = typemap.create_enum_typedef(self)
        self.typedef_name = self.typedef.name
        # also 'enum class foo' will alter scope

######################################################################

class TypedefNode(AstNode):
    """
    Used for namespace resolution
    """
    def __init__(self, name, parent):

        # From arguments
        self.name = name
        self.parent = parent

        # Add to namespace
        self.typename = self.parent.scope + self.name

    def get_typename(self):
        return self.typename

######################################################################

def create_std_namespace(glb):
    """Create the std namespace and add the types we care about.
    """
    std = glb.add_namespace('std')
    std.add_typedef('string')
    std.add_typedef('vector')
    return std

######################################################################
# Parse yaml file
######################################################################

def clean_dictionary(dd):
    """YAML converts some blank fields to None,
    but we want blank.
    """
    for key in ['cxx_header', 'namespace']:
        if key in dd and dd[key] is None:
            dd[key] = ''

    if 'default_arg_suffix' in dd:
        default_arg_suffix = dd['default_arg_suffix']
        if not isinstance(default_arg_suffix, list):
            raise RuntimeError('default_arg_suffix must be a list')
        for i, value in enumerate(dd['default_arg_suffix']):
            if value is None:
                dd['default_arg_suffix'][i] = ''

    if 'format' in dd:
        dd0 = dd['format']
        for key in ['function_suffix']:
            if key in dd0 and dd0[key] is None:
                dd0[key] = ''

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
            lst[i] = ''

def add_declarations(parent, node):
    if 'declarations' not in node:
        return

    for subnode in node['declarations']:
        if 'namespace' in subnode:
            name = subnode['namespace']
            d = copy.copy(subnode)
            clean_dictionary(d)
            del d['namespace']
            ns = parent.add_namespace(name, **d)
            add_declarations(ns, subnode)
        elif 'class' in subnode:
            name = subnode['class']
            d = copy.copy(subnode)
            clean_dictionary(d)
            del d['class']
            cls = parent.add_class(name, **d)
            add_declarations(cls, subnode)
        elif 'block' in subnode:
            d = copy.copy(subnode)
            clean_dictionary(d)
            blk = BlockNode(parent, **d)
            add_declarations(blk, subnode)
        elif 'decl' in subnode:
            # copy before clean to avoid changing input dict
            d = copy.copy(subnode)
            clean_dictionary(d)
            decl = d['decl']
            del d['decl']
            parent.add_declaration(decl, **d)
        elif 'forward' in subnode:
            key = subnode['forward']
            parent.add_typedef(key)
            if 'fields' in subnode:
                value = subnode['fields']
                if not isinstance(value, dict):
                    raise TypeError("fields must be a dictionary")
                fullname = parent.scope + key
                typedef = typemap.Typedef(fullname, **value)
                typedef.base = 'shadow'
                if 'cxx_type' not in value:
                    typedef.cxx_type = fullname
                typemap.typedef_shadow_defaults(typedef)
                typemap.Typedef.register(typedef.name, typedef)
        elif 'typedef' in subnode:
            key = subnode['typedef']
            value = subnode['fields']
            copy_type = value['typedef']
            def_types, def_types_alias = typemap.Typedef.get_global_types()
            orig = def_types.get(copy_type, None)
            if not orig:
                raise RuntimeError(
                    "No type for typedef {} while defining {}".format(copy_type, key))
            typedef = orig.clone_as(copy_type)
            typedef.update(value)
            typedef.name = parent.scope + key
            if 'cxx_type' not in value:
                typedef.cxx_type = typedef.name
            typemap.Typedef.register(typedef.name, typedef)
            parent.add_typedef(key)   # Add to namespace
        else:
            print(subnode)
            raise RuntimeError("Expected 'block', 'class', 'decl', 'forward', 'namespace' or 'typedef' found {}".format(sorted(subnode.keys())))

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

    if 'copyright' in node:
        clean_list(node['copyright'])

    clean_dictionary(node)
    library = LibraryNode(**node)

    add_declarations(library, node)

    return library
