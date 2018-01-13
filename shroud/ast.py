# Copyright (c) 2017, Lawrence Livermore National Security, LLC.
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
from . import typemap

class AstNode(object):
    def option_to_fmt(self):
        """Set fmt based on options dictionary.
        """
        for name in ['C_prefix', 'F_C_prefix', 
                     'C_this', 'C_result', 'CXX_this',
                     'F_this', 'F_result', 'F_derived_member',
                     'C_string_result_as_arg', 'F_string_result_as_arg',
                     'C_header_filename_suffix',
                     'C_impl_filename_suffix',
                     'F_filename_suffix',
                     'PY_header_filename_suffix',
                     'PY_impl_filename_suffix',
                     'PY_result',
                     'LUA_header_filename_suffix',
                     'LUA_impl_filename_suffix',
                     'LUA_result']:
            if self.options.inlocal(name):
                setattr(self._fmt, name, self.options[name])

    def eval_template(self, name, tname='', fmt=None):
        """fmt[name] = self.name or option[name + tname + '_template']
        """
        if fmt is None:
            fmt = self._fmt
        value = getattr(self, name)
        if value is not None:
            setattr(fmt, name, value)
        else:
            tname = name + tname + '_template'
            setattr(fmt, name, util.wformat(self.options[tname], fmt))

######################################################################

class LibraryNode(AstNode):
    def __init__(self,
                 cxx_header='',
                 language='c++',
                 library='default_library',
                 namespace='',
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
        self.namespace = namespace

        self.classes = []
        self.functions = []
        # Each is given a _function_index when created.
        self.function_index = []
        self.options = self.default_options()
        if options:
            self.options.update(options, replace=True)

        self.F_module_dependencies = []     # unused

        self.copyright = kwargs.setdefault('copyright', [])
        self.patterns = kwargs.setdefault('patterns', [])

        for n in ['C_header_filename', 'C_impl_filename',
                  'F_module_name', 'F_impl_filename',
                  'LUA_module_name', 'LUA_module_reg', 'LUA_module_filename', 'LUA_header_filename',
                  'PY_module_filename', 'PY_header_filename', 'PY_helper_filename',
                  'YAML_type_filename']:
            setattr(self, n, kwargs.get(n, None))

        self.default_format()
        self.option_to_fmt()

        # default some options based on other options
        self.eval_template('C_header_filename', '_library')
        self.eval_template('C_impl_filename', '_library')
        # All class/methods and functions may go into this file or
        # just functions.
        self.eval_template('F_module_name', '_library')
        self.eval_template('F_impl_filename', '_library')

    def default_options(self):
        """default options."""
        def_options = util.Options(
            parent=None,
            debug=False,   # print additional debug info

            F_module_per_class=True,
            F_string_len_trim=True,
            F_force_wrapper=False,

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

            C_name_template=(
                '{C_prefix}{class_prefix}{underscore_name}{function_suffix}'),

            C_bufferify_suffix='_bufferify',
            C_var_len_template = 'N{c_var}',         # argument for result of len(arg)
            C_var_trim_template = 'L{c_var}',        # argument for result of len_trim(arg)
            C_var_size_template = 'S{c_var}',        # argument for result of size(arg)

            # Fortran's names for C functions
            F_C_prefix='c_',
            F_C_name_template=(
                '{F_C_prefix}{class_prefix}{underscore_name}{function_suffix}'),

            F_name_impl_template=(
                '{class_prefix}{underscore_name}{function_suffix}'),

            F_name_function_template='{underscore_name}{function_suffix}',
            F_name_generic_template='{underscore_name}',

            F_module_name_library_template='{library_lower}_mod',
            F_impl_filename_library_template='wrapf{library_lower}.{F_filename_suffix}',

            F_module_name_class_template='{class_lower}_mod',
            F_impl_filename_class_template='wrapf{cxx_class}.{F_filename_suffix}',

            F_name_instance_get='get_instance',
            F_name_instance_set='set_instance',
            F_name_associated='associated',

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
                '{PY_prefix}{class_prefix}{underscore_name}{function_suffix}'),
            )
        return def_options

    def default_format(self):
        """Set format dictionary.
        """

        self._fmt = util.Options(None)
        fmt_library = self._fmt

        fmt_library.library = self.library
        fmt_library.library_lower = fmt_library.library.lower()
        fmt_library.library_upper = fmt_library.library.upper()
        fmt_library.function_suffix = ''   # assume no suffix
        fmt_library.C_prefix = self.options.get(
            'C_prefix', fmt_library.library_upper[:3] + '_')
        fmt_library.F_C_prefix = self.options['F_C_prefix']
        if self.namespace:
            fmt_library.namespace_scope = (
                '::'.join(self.namespace.split()) + '::')
        else:
            fmt_library.namespace_scope = ''

        # set default values for fields which may be unset.
        fmt_library.class_prefix = ''
#        fmt_library.c_ptr = ''
#        fmt_library.c_const = ''
        fmt_library.CXX_this_call = ''
        fmt_library.CXX_template = ''
        fmt_library.C_pre_call = ''
        fmt_library.C_post_call = ''

        fmt_library.C_this = 'self'
        fmt_library.C_result = 'SHT_rv'
        fmt_library.c_temp = 'SHT_'

        fmt_library.CXX_this = 'SH_this'

        fmt_library.F_this = 'obj'
        fmt_library.F_result = 'SHT_rv'
        fmt_library.F_derived_member = 'voidptr'

        fmt_library.C_string_result_as_arg = 'SHF_rv'
        fmt_library.F_string_result_as_arg = ''

        fmt_library.F_filename_suffix = 'f'

        # don't have to worry about argument names in Python wrappers
        # so skip the SH_ prefix by default.
        fmt_library.PY_result = 'rv'
        fmt_library.LUA_result = 'rv'

        if self.language == 'c':
            fmt_library.C_header_filename_suffix = 'h'
            fmt_library.C_impl_filename_suffix = 'c'

            fmt_library.PY_header_filename_suffix = 'h'
            fmt_library.PY_impl_filename_suffix = 'c'

            fmt_library.LUA_header_filename_suffix = 'h'
            fmt_library.LUA_impl_filename_suffix = 'c'

            fmt_library.stdlib  = ''
        else:
            fmt_library.C_header_filename_suffix = 'h'
            fmt_library.C_impl_filename_suffix = 'cpp'

            fmt_library.PY_header_filename_suffix = 'hpp'
            fmt_library.PY_impl_filename_suffix = 'cpp'

            fmt_library.LUA_header_filename_suffix = 'hpp'
            fmt_library.LUA_impl_filename_suffix = 'cpp'

            fmt_library.stdlib  = 'std::'

    def add_function(self, parentoptions=None, **kwargs):
        """Add a function.
        """
        fcnnode = FunctionNode(self, parentoptions=parentoptions, **kwargs)
        self.functions.append(fcnnode)
        return fcnnode

    def add_class(self, name, **kwargs):
        """Add a class.
        """
        clsnode = ClassNode(name, self, **kwargs)
        self.classes.append(clsnode)
        return clsnode

    def _to_dict(self):
        """Convert to dictionary.
        Used by util.ExpandedEncoder.
        """
        d = dict(
            _fmt=self._fmt,
            options=self.options,
        )

        for key in [ 'classes', 'copyright', 'cxx_header',
                     'functions', 'language', 'namespace' ]:
            value = getattr(self,key)
            if value:
                d[key] = value

        return d

######################################################################

class ClassNode(AstNode):
    def __init__(self, name, parent,
                 cxx_header='',
                 namespace='',
                 options=None,
                 **kwargs):
        """Create ClassNode.
        """
        # From arguments
        self.name = name
        self.cxx_header = cxx_header
        self.namespace = namespace

        self.functions = []

        self.python = kwargs.get('python', {})

        for n in ['C_header_filename', 'C_impl_filename',
                  'F_derived_name', 'F_impl_filename', 'F_module_name',
                  'LUA_userdata_type', 'LUA_userdata_member', 'LUA_class_reg',
                  'LUA_metadata', 'LUA_ctor_name',
                  'PY_PyTypeObject', 'PY_PyObject', 'PY_type_filename',
                  'class_prefix', 'cpp_if']:
            setattr(self, n, kwargs.get(n, None))

        self.options = util.Options(parent=parent.options)
        if options:
            self.options.update(options, replace=True)

        self._fmt = util.Options(parent._fmt)
        fmt_class = self._fmt
        fmt_class.cxx_class = name
        fmt_class.class_lower = name.lower()
        fmt_class.class_upper = name.upper()
        self.eval_template('class_prefix')

        # Only one file per class for C.
        self.eval_template('C_header_filename', '_class')
        self.eval_template('C_impl_filename', '_class')

        if self.options.F_module_per_class:
            self.eval_template('F_module_name', '_class')
            self.eval_template('F_impl_filename', '_class')

    def add_function(self, parentoptions=None, **kwargs):
        """Add a function.
        """
        fcnnode = FunctionNode(self, parentoptions=parentoptions, **kwargs)
        self.functions.append(fcnnode)
        return fcnnode

    def _to_dict(self):
        """Convert to dictionary.
        Used by util.ExpandedEncoder.
        """
        d = dict(
            _fmt = self._fmt,
            cxx_header=self.cxx_header,
            methods=self.functions,
            name=self.name,
            options=self.options,
        )
        for key in ['namespace', 'python']:
            value = getattr(self,key)
            if value:
                d[key] = value
        for key in ['C_header_filename', 'C_impl_filename',
                    'F_derived_name', 'F_impl_filename', 'F_module_name']:
            value = getattr(self,key)
            if value is not None:
                d[key] = value
        return d


######################################################################

class FunctionNode(AstNode):
    """

    - decl:
      cxx_template:
        ArgType:
        - int
        - double


    _fmtfunc = Option()

    _fmtresult = {
       'fmtc': Option(_fmtfunc)
    }
    _fmtargs = {
      'arg1': {
        'fmtc': Option(_fmtfunc),
        'fmtf': Option(_fmtfunc)
      }
    }

    _decl            - generated declaration.
                       Includes computed attributes
    _function_index  - sequence number function,
                       used in lieu of a pointer
    _generated       - who generated this function
    _PTR_F_C_index   - Used by fortran wrapper to find index of
                       C function to call
    _PTR_C_CXX_index - Used by C wrapper to find index of C++ function
                       to call
    _subprogram      - subroutine or function

    """
    def __init__(self, parent,
                 decl=None,
                 parentoptions=None,
                 options=None,
                 **kwargs):
        self.options = util.Options(parent= parentoptions or parent.options)
        if options:
            self.options.update(options, replace=True)

        self._fmt = util.Options(parent._fmt)
        self.option_to_fmt()

        # working variables
        self._CXX_return_templated = False
        self._PTR_C_CXX_index = None
        self._PTR_F_C_index = None
        self._cxx_overload = None
        self._decl = None
        self._default_funcs = []         #  generated default value functions  (unused?)
        self._function_index = None
        self._error_pattern_suffix = ''
        self._fmtargs = {}
        self._fmtresult = {}
        self._function_index = None
        self._generated = False
        self._has_default_arg = False
        self._nargs = None
        self._overloaded = False
        self._subprogram = 'XXX-subprogram'

#        self.function_index = []

        # Move fields from kwargs into instance
        for n in [
                'C_code', 'C_error_pattern', 'C_name',
                'C_post_call', 'C_post_call_buf',
                'C_return_code', 'C_return_type',
                'F_C_name', 'F_code',
                'F_name_function', 'F_name_generic', 'F_name_impl',
                'LUA_name', 'LUA_name_impl',
                'PY_error_pattern', 'PY_name_impl',
                'cpp_if', 'docs', 'function_suffix', 'return_this']:
            setattr(self, n, kwargs.get(n, None))

        self.default_arg_suffix = kwargs.get('default_arg_suffix', [])
        self.cxx_template = kwargs.get('cxx_template', {})
        self.doxygen = kwargs.get('doxygen', {})
        self.fortran_generic = kwargs.get('fortran_generic', {})

        # referenced explicity (not via fmt)
        # C_code, C_return_code, C_return_type, F_code
        
        if not decl:
            raise RuntimeError("Missing decl")

        # parse decl and add to dictionary
        if isinstance(parent,ClassNode):
            cls_name = parent.name
        else:
            cls_name = None
        template_types = self.cxx_template.keys()

        self.decl = decl
        ast = declast.check_decl(decl,
                                 current_class=cls_name,
                                 template_types=template_types)
        self._ast = ast

        # add any attributes from YAML files to the ast
        if 'attrs' in kwargs:
            attrs = kwargs['attrs']
            if 'result' in attrs:
                ast.attrs.update(attrs['result'])
            for arg in ast.params:
                name = arg.name
                if name in attrs:
                    arg.attrs.update(attrs[name])
        # XXX - waring about unused fields in attrs
                                    
        if ast.params is None:
            # 'void foo' instead of 'void foo()'
            raise RuntimeError("Missing arguments:", ast.gen_decl())

        fmt_func = self._fmt
        fmt_func.function_name = ast.name
        fmt_func.underscore_name = util.un_camel(fmt_func.function_name)

    def _to_dict(self):
        """Convert to dictionary.
        Used by util.ExpandedEncoder.
        """
        d = dict(
            _ast=self._ast,
            _fmt=self._fmt,
            _function_index=self._function_index,
            decl=self.decl,
            options=self.options,
        )
        for key in ['cxx_template', 'default_arg_suffix', 'docs', 'doxygen', 
                    'fortran_generic', 'return_this',
                    'C_code', 'C_error_pattern', 'C_name',
                    'C_post_call', 'C_post_call_buf', 
                    'C_return_code', 'C_return_type',
                    'F_C_name', 'F_code', 'F_name_function', 'F_name_generic', 'F_name_impl',
                    'PY_error_pattern',
                    '_PTR_C_CXX_index', '_PTR_F_C_index',
                    '_CXX_return_templated',
                    '_cxx_overload', '_error_pattern_suffix',
                    '_decl', '_default_funcs', 
                    '_fmtargs', '_fmtresult',
                    '_generated', '_has_default_arg',
                    '_nargs', '_overloaded', '_subprogram']:
            value = getattr(self,key)
            if value:
                d[key] = value

        for key in ['function_suffix']:
            value = getattr(self,key)
            if value is not None:   # '' is OK
                d[key] = value
        return d

    def clone(self):
        """Create a copy of a function node to use with C++ template
        or changing result to argument.
        """
        # Shallow copy everything
        new = copy.copy(self)

        # new layer of Options
        new._fmt = util.Options(self._fmt)
        new.options = util.Options(self.options)
    
        # deep copy dictionaries
        new._ast = copy.deepcopy(self._ast)
        new._fmtargs = copy.deepcopy(self._fmtargs)
        new._fmtresult = copy.deepcopy(self._fmtresult)
    
        return new


def clean_dictionary(dd):
    """YAML converts some blank fields to None,
    but we want blank.
    """
    for key in ['cxx_header', 'namespace',
                'function_suffix']:
        if key in dd and dd[key] is None:
            dd[key] = ''

    if 'default_arg_suffix' in dd:
        default_arg_suffix = dd['default_arg_suffix']
        if not isinstance(default_arg_suffix, list):
            raise RuntimeError('default_arg_suffix must be a list')
        for i, value in enumerate(dd['default_arg_suffix']):
            if value is None:
                dd['default_arg_suffix'][i] = ''


def is_options_only(node):
    """Detect an options only node.

    functions:
    - options:
         a = b
    - decl:
      options:

    Return True if node only has options.
    """
    if len(node) != 1:
        return False
    if 'options' not in node:
        return False
    if not isinstance(node['options'], dict):
        raise TypeError("options must be a dictionary")
    return True

def add_functions(parent, functions):
    """ Add functions from list 'functions'.
    Look for 'options' only entries.

    functions = [
      {
        'options': 
      },{
        'decl': 'void func1()'
      },{
        'decl': 'void func2()'
      }
    ]

    Used with class methods and functions.
    """
    if not isinstance(functions, list):
        raise TypeError("functions must be a list")

    options = parent.options
    for node in functions:
        if is_options_only(node):
            options = util.Options(options, **node['options'])
        else:
            clean_dictionary(node)
            parent.add_function(parentoptions=options, **node)

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

    if 'types' in node:
        types_dict = node['types']
        if not isinstance(types_dict, dict):
            raise TypeError("types must be a dictionary")
        def_types, def_types_alias = typemap.Typedef.get_global_types()
        for key, value in types_dict.items():
            if not isinstance(value, dict):
                raise TypeError("types '%s' must be a dictionary" % key)
            declast.add_type(key)   # Add to parser

            if 'typedef' in value:
                copy_type = value['typedef']
                orig = def_types.get(copy_type, None)
                if not orig:
                    raise RuntimeError(
                        "No type for typedef {}".format(copy_type))
                def_types[key] = typemap.Typedef(key)
                def_types[key].update(def_types[copy_type]._to_dict())

            if key in def_types:
                def_types[key].update(value)
            else:
                def_types[key] = typemap.Typedef(key, **value)
            typemap.typedef_wrapped_defaults(def_types[key])

    clean_dictionary(node)
    library = LibraryNode(**node)

    if 'classes' in node:
        classes = node['classes']
        if not isinstance(classes, list):
            raise TypeError("classes must be a list")

        # Add all class types to parser first
        # Emulate forward declarations of classes.
        for cls in classes:
            if not isinstance(cls, dict):
                raise TypeError("classes[n] must be a dictionary")
            if 'name' not in cls:
                raise TypeError("class does not define name")
            clean_dictionary(cls)
            declast.add_type(cls['name'])

        for cls in classes:
            clsnode = library.add_class(**cls)
            if 'methods' in cls:
                add_functions(clsnode, cls['methods'])
            elif 'functions' in cls:
                add_functions(clsnode, cls['functions'])

    if 'functions' in node:
        add_functions(library, node['functions'])

    return library
