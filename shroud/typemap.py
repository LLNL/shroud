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
Create and manage typemaps used to convert between languages.
"""

from . import util

class Typedef(object):
    """ Collect fields for an argument.
    This used to be a dict but a class has better access semantics:
       i.attr vs d['attr']
    It also initializes default values to avoid  d.get('attr', default)
    """

    # Array of known keys with default values
    _order = (
        ('base', 'unknown'),      # Base type: 'string'
        ('forward', None),        # Forward declaration
        ('typedef', None),        # Initialize from existing type

        ('cxx_type', None),       # Name of type in C++
        ('cxx_to_c', None),       # Expression to convert from C++ to C
                                  # None implies {cxx_var} i.e. no conversion
        ('cxx_header', None),     # Name of C++ header file required for implementation
                                  # For example, if cxx_to_c was a function

        ('c_type', None),         # Name of type in C
        ('c_header', None),       # Name of C header file required for type
        ('c_to_cxx', None),       # Expression to convert from C to C++
                                  # None implies {c_var}  i.e. no conversion
        ('c_statements', {}),
        ('c_templates', {}),      # c_statements for cxx_T
        ('c_return_code', None),

        ('f_c_args', None),       # List of argument names to F_C routine
        ('f_c_argdecl', None),    # List of declarations to F_C routine
        ('f_c_module', None),     # Fortran modules needed for interface  (dictionary)

        ('f_type', None),         # Name of type in Fortran -- integer(C_INT)
        ('f_kind', None),         # Fortran kind            -- C_INT
        ('f_c_type', None),       # Type for C interface    -- int
        ('f_to_c', None),         # Expression to convert from Fortran to C
        ('f_derived_type', None), # Fortran derived type name
        ('f_args', None),         # Argument in Fortran wrapper to call C.
        ('f_module', None),       # Fortran modules needed for type  (dictionary)
        ('f_cast', '{f_var}'),    # Expression to convert to type
                                  # e.g. intrinsics such as int and real
        ('f_statements', {}),

        ('result_as_arg', None),  # override fields when result should be treated as an argument

        # Python
        ('PY_format', 'O'),       # 'format unit' for PyArg_Parse
        ('PY_PyTypeObject', None), # variable name of PyTypeObject instance
        ('PY_PyObject', None),    # typedef name of PyObject instance
        ('PY_ctor', None),        # expression to create object.
                                  # ex. PyBool_FromLong({rv})
        ('PY_to_object', None),   # PyBuild - object'=converter(address)
        ('PY_from_object', None), # PyArg_Parse - status=converter(object, address);
        ('PY_build_arg', None),   # argument for Py_BuildValue
        ('py_statements', {}),

        # Lua
        ('LUA_type', 'LUA_TNONE'),
        ('LUA_pop', 'POP'),
        ('LUA_push', 'PUSH'),
        ('LUA_statements', {}),
    )


    _keyorder, _valueorder = zip(*_order)

    # valid fields
    defaults = dict(_order)

    def __init__(self, name, **kw):
        self.name = name
#        for key, defvalue in self.defaults.items():
#            setattr(self, key, defvalue)
        self.__dict__.update(self.defaults)  # set all default values
        self.update(kw)

    def update(self, d):
        """Add options from dictionary to self.
        """
        for key in d:
            if key in self.defaults:
                setattr(self, key, d[key])
            else:
                raise RuntimeError("Unknown key for Argument %s", key)

    def XXXcopy(self):
        n = Typedef(self.name)
        n.update(self._to_dict())
        return n

    def clone_as(self, name):
        n = Typedef(name)
        n.update(self._to_dict())
        return n

    def _to_dict(self):
        """Convert instance to a dictionary for json.
        """
        # only export non-default values
        a = {}
        for key, defvalue in self.defaults.items():
            value = getattr(self, key)
            if value is not defvalue:
                a[key] = value
        return a

    def __repr__(self):
        # only print non-default values
        args = []
        for key, defvalue in self.defaults.items():
            value = getattr(self, key)
            if value is not defvalue:
                if isinstance(value, str):
                    args.append("{0}='{1}'".format(key, value))
                else:
                    args.append("{0}={1}".format(key, value))
        return "Typedef('%s', " % self.name + ','.join(args) + ')'

    def __as_yaml__(self, indent, output):
        """Write out entire typedef as YAML.
        """
        util.as_yaml(self, self._keyorder, indent, output)

    def __export_yaml__(self, indent, output):
        """Write out a subset of a wrapped type.
        Other fields are set with typedef_shadow_defaults.
        """
        util.as_yaml(self, [
            'base',
            'cxx_header',
            'cxx_type',
            'c_type',
            'c_header',
            'f_derived_type',
            'f_to_c',
            'f_module',
        ], indent, output)


    ### Manage collection of typedefs
    _typedict = {}   # dictionary of registered types
    _typealias = {}  # dictionary of registered type aliases
    @classmethod
    def set_global_types(cls, typedict, typealias):
        cls._typedict = typedict
        cls._typealias = typealias

    @classmethod
    def get_global_types(cls):
        return cls._typedict, cls._typealias

    @classmethod
    def register(cls, name, typedef):
        """Register a typedef"""
        cls._typedict[name] = typedef

    @classmethod
    def lookup(cls, name):
        """Lookup name in registered types taking aliases into account."""
        typedef = cls._typedict.get(cls._typealias.get(name,name), None)
        return typedef

    @classmethod
    def resolve_alias(cls, name):
        """return typedef for alias.
        """
        return cls._typealias.get(name, name)


def initialize():
    def_types = dict(
        void=Typedef(
            'void',
            c_type='void',
            cxx_type='void',
            # fortran='subroutine',
            f_type='type(C_PTR)',
            f_module=dict(iso_c_binding=['C_PTR']),
            PY_ctor='PyCapsule_New({cxx_var}, NULL, NULL)',
            ),
        int=Typedef(
            'int',
            c_type='int',
            cxx_type='int',
            f_cast='int({f_var}, C_INT)',
            f_type='integer(C_INT)',
            f_kind='C_INT',
            f_module=dict(iso_c_binding=['C_INT']),
            PY_format='i',
            PY_ctor='PyInt_FromLong({c_var})',
            LUA_type='LUA_TNUMBER',
            LUA_pop='lua_tointeger({LUA_state_var}, {LUA_index})',
            LUA_push='lua_pushinteger({LUA_state_var}, {c_var})',
            ),
        long=Typedef(
            'long',
            c_type='long',
            cxx_type='long',
            f_cast='int({f_var}, C_LONG)',
            f_type='integer(C_LONG)',
            f_kind='C_LONG',
            f_module=dict(iso_c_binding=['C_LONG']),
            PY_format='l',
            PY_ctor='PyInt_FromLong({c_var})',
            LUA_type='LUA_TNUMBER',
            LUA_pop='lua_tointeger({LUA_state_var}, {LUA_index})',
            LUA_push='lua_pushinteger({LUA_state_var}, {c_var})',
            ),
        long_long=Typedef(
            'long_long',
            c_type='long long',
            cxx_type='long long',
            f_cast='int({f_var}, C_LONG_LONG)',
            f_type='integer(C_LONG_LONG)',
            f_kind='C_LONG_LONG',
            f_module=dict(iso_c_binding=['C_LONG_LONG']),
            PY_format='L',
#            PY_ctor='PyInt_FromLong({c_var})',
            LUA_type='LUA_TNUMBER',
            LUA_pop='lua_tointeger({LUA_state_var}, {LUA_index})',
            LUA_push='lua_pushinteger({LUA_state_var}, {c_var})',
            ),
        size_t=Typedef(
            'size_t',
            c_type='size_t',
            cxx_type='size_t',
            c_header='stdlib.h',
            f_cast='int({f_var}, C_SIZE_T)',
            f_type='integer(C_SIZE_T)',
            f_kind='C_SIZE_T',
            f_module=dict(iso_c_binding=['C_SIZE_T']),
            PY_ctor='PyInt_FromSize_t({c_var})',
            LUA_type='LUA_TNUMBER',
            LUA_pop='lua_tointeger({LUA_state_var}, {LUA_index})',
            LUA_push='lua_pushinteger({LUA_state_var}, {c_var})',
            ),

        float=Typedef(
            'float',
            c_type='float',
            cxx_type='float',
            f_cast='real({f_var}, C_FLOAT)',
            f_type='real(C_FLOAT)',
            f_kind='C_FLOAT',
            f_module=dict(iso_c_binding=['C_FLOAT']),
            PY_format='f',
            PY_ctor='PyFloat_FromDouble({c_var})',
            LUA_type='LUA_TNUMBER',
            LUA_pop='lua_tonumber({LUA_state_var}, {LUA_index})',
            LUA_push='lua_pushnumber({LUA_state_var}, {c_var})',
            ),
        double=Typedef(
            'double',
            c_type='double',
            cxx_type='double',
            f_cast='real({f_var}, C_DOUBLE)',
            f_type='real(C_DOUBLE)',
            f_kind='C_DOUBLE',
            f_module=dict(iso_c_binding=['C_DOUBLE']),
            PY_format='d',
            PY_ctor='PyFloat_FromDouble({c_var})',
            LUA_type='LUA_TNUMBER',
            LUA_pop='lua_tonumber({LUA_state_var}, {LUA_index})',
            LUA_push='lua_pushnumber({LUA_state_var}, {c_var})',
            ),

        bool=Typedef(
            'bool',
            c_type='bool',
            cxx_type='bool',

            f_type='logical',
            f_kind='C_BOOL',
            f_c_type='logical(C_BOOL)',
            f_module=dict(iso_c_binding=['C_BOOL']),

            f_statements=dict(
                intent_in=dict(
                    c_local_var=True,
                    pre_call=[
                        '{c_var} = {f_var}  ! coerce to C_BOOL',
                        ],
                    ),
                intent_out=dict(
                    c_local_var=True,
                    post_call=[
                        '{f_var} = {c_var}  ! coerce to logical',
                        ],
                    ),
                intent_inout=dict(
                    c_local_var=True,
                    pre_call=[
                        '{c_var} = {f_var}  ! coerce to C_BOOL',
                        ],
                    post_call=[
                        '{f_var} = {c_var}  ! coerce to logical',
                        ],
                    ),
                result=dict(
                    # The wrapper is needed to convert bool to logical
                    need_wrapper=True,
                    ),
                ),

            py_statements=dict(
                intent_in=dict(
                    pre_call=[
                        'bool {cxx_var} = PyObject_IsTrue({py_var});',
                    ],
                ),
                intent_inout=dict(
                    pre_call=[
                        'bool {cxx_var} = PyObject_IsTrue({py_var});',
                    ],
                    # py_var is already declared for inout
                    post_call=[
                        '{py_var} = PyBool_FromLong({c_var});',
                    ],
                ),
                intent_out=dict(
                    post_call=[
                        '{PyObject} * {py_var} = PyBool_FromLong({c_var});',
                    ],
                ),
            ),

            # XXX PY_format='p',  # Python 3.3 or greater
# Use py_statements.x.ctor instead of PY_ctor. This code will always be
# added.  Older version of Python can not create a bool directly from
# from Py_BuildValue.
#            PY_ctor='PyBool_FromLong({c_var})',
            PY_PyTypeObject='PyBool_Type',

            LUA_type='LUA_TBOOLEAN',
            LUA_pop='lua_toboolean({LUA_state_var}, {LUA_index})',
            LUA_push='lua_pushboolean({LUA_state_var}, {c_var})',
            ),

        # implies null terminated string
        char=Typedef(
            'char',
            cxx_type='char',

            c_type='char',    # XXX - char *

            c_statements=dict(
                intent_in_buf=dict(
                    buf_args = [ 'len_trim' ],
                    cxx_local_var='pointer',
                    c_header='<stdlib.h> <string.h>',
                    cxx_header='<stdlib.h> <cstring>',
                    pre_call=[
                        'char * {cxx_var} = (char *) malloc({c_var_trim} + 1);',
                        '{stdlib}memcpy({cxx_var}, {c_var}, {c_var_trim});',
                        '{cxx_var}[{c_var_trim}] = \'\\0\';'
                        ],
                    post_call=[
                        'free({cxx_var});'
                        ],
                    ),
                intent_out_buf=dict(
                    buf_args = [ 'len' ],
                    cxx_local_var='pointer',
                    c_header='<stdlib.h>',
                    cxx_header='<stdlib.h>',
                    c_helper='ShroudStrCopy',
                    pre_call=[
                        'char * {cxx_var} = (char *) malloc({c_var_len} + 1);',
                        ],
                    post_call=[
                        'ShroudStrCopy({c_var}, {c_var_len}, {cxx_var});',
                        'free({cxx_var});',
                        ],
                    ),
                intent_inout_buf=dict(
                    buf_args = [ 'len_trim', 'len' ],
                    cxx_local_var='pointer',
                    c_helper='ShroudStrCopy',
                    c_header='<stdlib.h> <string.h>',
                    cxx_header='<stdlib.h> <cstring>',
                    pre_call=[
                        'char * {cxx_var} = (char *) malloc({c_var_len} + 1);',
                        '{stdlib}memcpy({cxx_var}, {c_var}, {c_var_trim});',
                        '{cxx_var}[{c_var_trim}] = \'\\0\';'
                        ],
                    post_call=[
                        'ShroudStrCopy({c_var}, {c_var_len}, {cxx_var});',
                        'free({cxx_var});',
                        ],
                    ),
                result_buf=dict(
                    buf_args = [ 'len' ],
                    c_header='<string.h>',
                    cxx_header='<cstring>',
                    c_helper='ShroudStrCopy',
                    post_call=[
                        'if ({cxx_var} == NULL) {{',
                        '    {stdlib}memset({c_var}, \' \', {c_var_len});',
                        '}} else {{',
                        '    ShroudStrCopy({c_var}, {c_var_len}, {cxx_var});',
                        '}}',
                        ],
                    ),
                ),

            f_type='character(*)',
            f_kind='C_CHAR',
            f_c_type='character(kind=C_CHAR)',
            f_c_module=dict(iso_c_binding=['C_CHAR']),

            f_statements=dict(
                result_pure=dict(
                    need_wrapper=True,
                    f_helper='fstr_ptr',
                    call=[
                        '{F_result} = fstr_ptr({F_C_call}({F_arg_c_call}))',
                        ],
                    )
                ),

            PY_format='s',
            PY_ctor='PyString_FromString({c_var})',
            LUA_type='LUA_TSTRING',
            LUA_pop='lua_tostring({LUA_state_var}, {LUA_index})',
            LUA_push='lua_pushstring({LUA_state_var}, {c_var})',
            base='string',
            ),

        # char scalar
        char_scalar=Typedef(
            'char_scalar',
            cxx_type='char',

            c_type='char',    # XXX - char *

            c_statements=dict(
                result_buf=dict(
                    buf_args = [ 'len' ],
                    c_header='<string.h>',
                    cxx_header='<cstring>',
                    post_call=[
                        '{stdlib}memset({c_var}, \' \', {c_var_len});',
                        '{c_var}[0] = {cxx_var};',
                    ],
                ),
            ),

            f_type='character',
            f_kind='C_CHAR',
            f_c_type='character(kind=C_CHAR)',
            f_c_module=dict(iso_c_binding=['C_CHAR']),
            PY_format='c',
#            PY_ctor='Py_BuildValue("c", (int) {c_var})',
            PY_ctor='PyString_FromStringAndSize(&{c_var}, 1)',
#            PY_build_format='c',
            PY_build_arg='(int) {cxx_var}',

            LUA_type='LUA_TSTRING',
            LUA_pop='lua_tostring({LUA_state_var}, {LUA_index})',
            LUA_push='lua_pushstring({LUA_state_var}, {c_var})',
            # # base='string',
            ),

        # C++ std::string
        string=Typedef(
            'string',
            cxx_type='std::string',
            cxx_header='<string>',
            cxx_to_c='{cxx_var}{cxx_deref}c_str()',  # cxx_deref is . or ->

            c_type='char',    # XXX - char *

            c_statements=dict(
                intent_in=dict(
                    cxx_local_var='scalar',
                    pre_call=[
                        '{c_const}std::string {cxx_var}({c_var});'
                        ],
                ),
                intent_out=dict(
                    cxx_header='<cstring>',
#                    pre_call=[
#                        'int {c_var_trim} = strlen({c_var});',
#                        ],
                    cxx_local_var='scalar',
                    pre_call=[
                        '{c_const}std::string {cxx_var};'
                        ],
                    post_call=[
                        # This may overwrite c_var if cxx_val is too long
                        'strcpy({c_var}, {cxx_var}{cxx_deref}c_str());'
                    ],
                ),
                intent_inout=dict(
                    cxx_header='<cstring>',
                    cxx_local_var='scalar',
                    pre_call=[
                        '{c_const}std::string {cxx_var}({c_var});'
                        ],
                    post_call=[
                        # This may overwrite c_var if cxx_val is too long
                        'strcpy({c_var}, {cxx_var}{cxx_deref}c_str());'
                    ],
                ),
                intent_in_buf=dict(
                    buf_args = [ 'len_trim' ],
                    cxx_local_var='scalar',
                    pre_call=[
                        ('{c_const}std::string '
                         '{cxx_var}({c_var}, {c_var_trim});')
                    ],
                ),
                intent_out_buf=dict(
                    buf_args = [ 'len' ],
                    c_helper='ShroudStrCopy',
                    cxx_local_var='scalar',
                    pre_call=[
                        'std::string {cxx_var};'
                    ],
                    post_call=[
                        'ShroudStrCopy({c_var}, {c_var_len}, {cxx_var}{cxx_deref}c_str());'
                    ],
                ),
                intent_inout_buf=dict(
                    buf_args = [ 'len_trim', 'len' ],
                    c_helper='ShroudStrCopy',
                    cxx_local_var='scalar',
                    pre_call=[
                        'std::string {cxx_var}({c_var}, {c_var_trim});'
                    ],
                    post_call=[
                        'ShroudStrCopy({c_var}, {c_var_len},'
                        '\t {cxx_var}{cxx_deref}c_str());'
                    ],
                ),
                result_buf=dict(
                    buf_args = [ 'len' ],
                    cxx_header='<cstring>',
                    c_helper='ShroudStrCopy',
                    post_call=[
                        'if ({cxx_var}{cxx_deref}empty()) {{',
                        '    {stdlib}memset({c_var}, \' \', {c_var_len});',
                        '}} else {{',
                        '    ShroudStrCopy({c_var}, {c_var_len}, {cxx_var}{cxx_deref}c_str());',
                        '}}',
                    ],
                ),
            ),

            f_type='character(*)',
            f_kind='C_CHAR',
            f_c_type='character(kind=C_CHAR)',
            f_c_module=dict(iso_c_binding=['C_CHAR']),

            f_statements=dict(
                result_pure=dict(
                    need_wrapper=True,
                    f_helper='fstr_ptr',
                    call=[
                        '{F_result} = fstr_ptr({F_C_call}({F_arg_c_call}))',
                        ],
                    )
                ),

            py_statements=dict(
                intent_in=dict(
                    cxx_local_var='scalar',
                    post_parse=[
                        '{c_const}std::string {cxx_var}({c_var});'
                    ],
                ),
                intent_inout=dict(
                    cxx_local_var='scalar',
                    post_parse=[
                        '{c_const}std::string {cxx_var}({c_var});'
                    ],
                ),
                intent_out=dict(
                    cxx_local_var='scalar',
                    post_parse=[
                        '{c_const}std::string {cxx_var};'
                    ],
                ),
            ),
            PY_format='s',
            PY_ctor='PyString_FromString({cxx_var}{cxx_deref}c_str())',
            PY_build_arg='{cxx_var}{cxx_deref}c_str()',

            LUA_type='LUA_TSTRING',
            LUA_pop='lua_tostring({LUA_state_var}, {LUA_index})',
            LUA_push='lua_pushstring({LUA_state_var}, {c_var})',
            base='string',
            ),

        # C++ std::string
        # Uses a two part call to copy results of std::string into a 
        # allocatable Fortran array.
        #    c_step1(stringout **out, int lenout)
        #    allocate(character(len=lenout): Fout)
        #    c_step2(Fout, out)
        # only used with bufferifed routines and intent(out) or result
        stringout=Typedef(
            'stringout',
            cxx_type='std::string',
            cxx_header='<string>',
            cxx_to_c='static_cast<void *>({cxx_var})',

            c_type='void',

            c_statements=dict(
#--                intent_in=dict(
#--                    cxx_local_var='scalar',
#--                    pre_call=[
#--                        '{c_const}std::string {cxx_var}({c_var});'
#--                        ],
#--                ),
#--                intent_out=dict(
#--                    cxx_header='<cstring>',
#--#                    pre_call=[
#--#                        'int {c_var_trim} = strlen({c_var});',
#--#                        ],
#--                    cxx_local_var='scalar',
#--                    pre_call=[
#--                        '{c_const}std::string {cxx_var};'
#--                        ],
#--                    post_call=[
#--                        # This may overwrite c_var if cxx_val is too long
#--                        'strcpy({c_var}, {cxx_var}{cxx_deref}c_str());'
#--                    ],
#--                ),
#--                intent_inout=dict(
#--                    cxx_header='<cstring>',
#--                    cxx_local_var='scalar',
#--                    pre_call=[
#--                        '{c_const}std::string {cxx_var}({c_var});'
#--                        ],
#--                    post_call=[
#--                        # This may overwrite c_var if cxx_val is too long
#--                        'strcpy({c_var}, {cxx_var}{cxx_deref}c_str());'
#--                    ],
#--                ),
#--                intent_in_buf=dict(
#--                    buf_args = [ 'len_trim' ],
#--                    cxx_local_var='scalar',
#--                    pre_call=[
#--                        ('{c_const}std::string '
#--                         '{cxx_var}({c_var}, {c_var_trim});')
#--                    ],
#--                ),
                intent_out_buf=dict(
                    buf_args = [ 'lenout' ],
                    c_helper='copy_string',
                    cxx_local_var='scalar',
                    pre_call=[
                        'std::string * {cxx_var};'
                    ],
                    post_call=[
                        ' post_call intent_out_buf'
                    ],
                ),
                result_buf=dict(
                    # pass address of string and length back to Fortran
                    buf_args = [ 'lenout' ],
                    c_helper='copy_string',
                    # Copy address of result into c_var and save length.
                    # When returning a std::string (and not a reference or pointer)
                    # an intermediate object is created to save the results
                    # which will be passed to copy_string
                    post_call=[
                        '*{c_var} = {cxx_addr}{cxx_var};',
                        '*{c_var_len} = {cxx_var}{cxx_deref}size();',
                    ],
                ),
            ),

            f_type='type(C_PTR)YY',
#            f_kind='C_CHAR',
            f_c_type='type(C_PTR)',
            f_c_module=dict(iso_c_binding=['C_PTR']),

            f_statements=dict(
                result_buf=dict(
                    need_wrapper=True,
                    f_helper='copy_string',
                    post_call=[
                        'allocate(character(len={f_var_len}, kind=C_CHAR):: {f_var})',
                        'call SHROUD_string_copy_and_free({f_cptr}, {f_var})',
                        ],
                    )
                ),

            base='string',
            ),

        # C++ std::vector
        # No c_type or f_type, use attr[template]
        vector=Typedef(
            'vector',
            cxx_type='std::vector<{cxx_T}>',
            cxx_header='<vector>',
#            cxx_to_c='{cxx_var}.data()',  # C++11

            c_statements=dict(
                intent_in_buf=dict(
                    buf_args = [ 'size' ],
                    cxx_local_var='scalar',
                    pre_call=[
                        ('{c_const}std::vector<{cxx_T}> '
                         '{cxx_var}({c_var}, {c_var} + {c_var_size});')
                    ],
                ),
                intent_out_buf=dict(
                    buf_args = [ 'size' ],
                    cxx_local_var='scalar',
                    pre_call=[
                        '{c_const}std::vector<{cxx_T}>'
                        '\t {cxx_var}({c_var_size});'
                    ],
                    post_call=[
                        '{{',
                        '    std::vector<{cxx_T}>::size_type',
                        '        {c_temp}i = 0,',
                        '        {c_temp}n = {c_var_size};',
                        '    {c_temp}n = std::min({cxx_var}.size(), {c_temp}n);',
                        '    for(; {c_temp}i < {c_temp}n; {c_temp}i++) {{',
                        '        {c_var}[{c_temp}i] = {cxx_var}[{c_temp}i];',
                        '    }}',
                        '}}'
                    ],
                ),
                intent_inout_buf=dict(
                    buf_args = [ 'size' ],
                    cxx_local_var='scalar',
                    pre_call=[
                        'std::vector<{cxx_T}> {cxx_var}('
                        '\t{c_var}, {c_var} + {c_var_size});'
                    ],
                    post_call=[
                        '{{+',
                        'std::vector<{cxx_T}>::size_type+',
                        '{c_temp}i = 0,',
                        '{c_temp}n = {c_var_size};',
                        '-{c_temp}n = std::min({cxx_var}.size(), {c_temp}n);',
                        'for(; {c_temp}i < {c_temp}n; {c_temp}i++) {{+',
                        '{c_var}[{c_temp}i] = {cxx_var}[{c_temp}i];',
                        '-}}',
                        '-}}'
                    ],
                ),
#                result_buf=dict(
#                    buf_args = [ 'size' ],
#                    c_helper='ShroudStrCopy',
#                    post_call=[
#                        'if ({cxx_var}.empty()) {{',
#                        '  std::memset({c_var}, \' \', {c_var_len});',
#                        '}} else {{',
#                        '  ShroudStrCopy({c_var}, {c_var_len}, {cxx_var}{cxx_deref}c_str());',
#                        '}}',
#                    ],
#                ),
            ),

#
            # custom code for templates
            c_templates=dict(
                string=dict(
                    intent_in_buf=dict(
                        buf_args = [ 'size', 'len' ],
                        c_helper='ShroudLenTrim',
                        cxx_local_var='scalar',
                        pre_call=[
                            'std::vector<{cxx_T}> {cxx_var};',
                            '{{',
                            '      {c_const}char * BBB = {c_var};',
                            '      std::vector<{cxx_T}>::size_type',
                            '        {c_temp}i = 0,',
                            '        {c_temp}n = {c_var_size};',
                            '    for(; {c_temp}i < {c_temp}n; {c_temp}i++) {{',
                            '        {cxx_var}.push_back(std::string(BBB,ShroudLenTrim(BBB, {c_var_len})));',
                            '        BBB += {c_var_len};',
                            '    }}',
                            '}}'
                        ],
                    ),
                    intent_out_buf=dict(
                        buf_args = [ 'size', 'len' ],
                        c_helper='ShroudLenTrim',
                        cxx_local_var='scalar',
                        pre_call=[
                            '{c_const}std::vector<{cxx_T}> {cxx_var};'
                        ],
                        post_call=[
                            '{{',
                            '    char * BBB = {c_var};',
                            '    std::vector<{cxx_T}>::size_type',
                            '        {c_temp}i = 0,',
                            '        {c_temp}n = {c_var_size};',
                            '    {c_temp}n = std::min({cxx_var}.size(),{c_temp}n);',
                            '    for(; {c_temp}i < {c_temp}n; {c_temp}i++) {{',
                            '        ShroudStrCopy(BBB, {c_var_len}, {cxx_var}[{c_temp}i].c_str());',
                            '        BBB += {c_var_len};',
                            '    }}',
                            '}}'
                        ],
                    ),
                    intent_inout_buf=dict(
                        buf_args = [ 'size', 'len' ],
                        cxx_local_var='scalar',
                        pre_call=[
                            'std::vector<{cxx_T}> {cxx_var};',
                            '{{',
                            '    {c_const}char * BBB = {c_var};',
                            '    std::vector<{cxx_T}>::size_type',
                            '        {c_temp}i = 0,',
                            '        {c_temp}n = {c_var_size};',
                            '    for(; {c_temp}i < {c_temp}n; {c_temp}i++) {{',
                            '        {cxx_var}.push_back(std::string(BBB,ShroudLenTrim(BBB, {c_var_len})));',
                            '        BBB += {c_var_len};',
                            '    }}',
                            '}}'
                        ],
                        post_call=[
                            '{{',
                            '    char * BBB = {c_var};',
                            '    std::vector<{cxx_T}>::size_type',
                            '        {c_temp}i = 0,',
                            '        {c_temp}n = {c_var_size};',
                            '    {c_temp}n = std::min({cxx_var}.size(),{c_temp}n);',
                            '    for(; {c_temp}i < {c_temp}n; {c_temp}i++) {{',
                            '        ShroudStrCopy(BBB, {c_var_len}, {cxx_var}[{c_temp}i].c_str());',
                            '        BBB += {c_var_len};',
                            '    }}',
                            '}}'
                        ],
                    ),
#                    result_buf=dict(
#                        c_helper='ShroudStrCopy',
#                        post_call=[
#                            'if ({cxx_var}.empty()) {{',
#                            '  std::memset({c_var}, \' \', {c_var_len});',
#                            '}} else {{',
#                            '  ShroudStrCopy({c_var}, {c_var_len}, {cxx_var}{cxx_deref}c_str());',
#                            '}}',
#                        ],
#                    ),
                ),
            ),
#



#            py_statements=dict(
#                intent_in=dict(
#                    cxx_local_var=True,
#                    post_parse=[
#                        '{c_const}std::vector<{cxx_T}> {cxx_var}({c_var});'
#                        ],
#                    ),
#                ),
#            PY_format='s',
#            PY_ctor='PyString_FromString({c_var})',
#            LUA_type='LUA_TSTRING',
#            LUA_pop='lua_tostring({LUA_state_var}, {LUA_index})',
#            LUA_push='lua_pushstring({LUA_state_var}, {c_var})',
            base='vector',
            ),

        MPI_Comm=Typedef(
            'MPI_Comm',
            cxx_type='MPI_Comm',
            c_header='mpi.h',
            c_type='MPI_Fint',
            # usually, MPI_Fint will be equivalent to int
            f_type='integer',
            f_kind='C_INT',
            f_c_type='integer(C_INT)',
            f_c_module=dict(iso_c_binding=['C_INT']),
            cxx_to_c='MPI_Comm_c2f({cxx_var})',
            c_to_cxx='MPI_Comm_f2c({c_var})',
            ),
        )

    # aliases
    def_types_alias = dict()
    def_types_alias['std::string'] = 'string'
    def_types_alias['std::vector'] = 'vector'
    def_types_alias['integer(C_INT)'] = 'int'
    def_types_alias['integer(C_LONG)'] = 'long'
    def_types_alias['integer(C_LONG_LONG)'] = 'long_long'
    def_types_alias['real(C_FLOAT)'] = 'float'
    def_types_alias['real(C_DOUBLE)'] = 'double'

    Typedef.set_global_types(def_types, def_types_alias)

    return def_types, def_types_alias


def create_class_typedef(cls):
    name = cls.name
    fmt_class = cls.fmtdict

    typedef = Typedef.lookup(name)
    if typedef is None:
        # unname = util.un_camel(name)
        unname = name.lower()
        cname = fmt_class.C_prefix + unname
        typedef = Typedef(
            name,
            base='shadow',
            cxx_type=fmt_class.namespace_scope + name,
            c_type=cname,
            f_derived_type=fmt_class.F_derived_name,
            f_module={fmt_class.F_module_name:[unname]},
            f_to_c = '{f_var}%%%s()' % fmt_class.F_name_instance_get,
            )
        typedef_shadow_defaults(typedef)
        Typedef.register(name, typedef)

    fmt_class.C_type_name = typedef.c_type


def typedef_shadow_defaults(typedef):
    """Add some defaults to typedef.
    When dumping typedefs to a file, only a subset is written
    since the rest are boilerplate.  This function restores
    the boilerplate.
    """
    if typedef.base != 'shadow':
        return

    typedef.cxx_to_c=('\tstatic_cast<{c_const}%s *>('
                      '\tstatic_cast<{c_const}void *>(\t{cxx_var}))' %
                      typedef.c_type)

    # opaque pointer -> void pointer -> class instance pointer
    typedef.c_to_cxx=('\tstatic_cast<{c_const}%s *>('
                      '\tstatic_cast<{c_const}void *>(\t{c_var}))' %
                      typedef.cxx_type)

    typedef.f_type='type(%s)' % typedef.f_derived_type
    typedef.f_c_type='type(C_PTR)'

    # XXX module name may not conflict with type name
#    typedef.f_module={fmt_class.F_module_name:[unname]}

    # return from C function
    # f_c_return_decl='type(CPTR)' % unname,
    typedef.f_statements = dict(
        result=dict(
            need_wrapper=True,
            call=[
                ('{F_result}%{F_derived_member} = '
                 '{F_C_call}({F_arg_c_call})')
                ],
            )
        )
    typedef.f_c_module={ 'iso_c_binding': ['C_PTR']}

    typedef.py_statements=dict(
        intent_in=dict(
            cxx_local_var='pointer',
            post_parse=[
                '{c_const}%s * {cxx_var} = '
                '{py_var} ? {py_var}->{PY_obj} : NULL;' % typedef.cxx_type,
            ],
        ),
        intent_inout=dict(
            cxx_local_var='pointer',
            post_parse=[
                '{c_const}%s * {cxx_var} = '
                '{py_var} ? {py_var}->{PY_obj} : NULL;' % typedef.cxx_type,
            ],
        ),
        intent_out=dict(
            post_call=[
                ('{PyObject} * {py_var} = '
                 'PyObject_New({PyObject}, &{PyTypeObject});'),
                '{py_var}->{PY_obj} = {cxx_var};',
            ]
        ),
    )
#    if not typedef.PY_PyTypeObject:
#        typedef.PY_PyTypeObject='UUU'
    # typedef.PY_ctor='PyObject_New({PyObject}, &{PyTypeObject})'

    typedef.LUA_type='LUA_TUSERDATA'
    typedef.LUA_pop=('\t({LUA_userdata_type} *)\t luaL_checkudata'
                     '(\t{LUA_state_var}, 1, "{LUA_metadata}")')
    # typedef.LUA_push=None  # XXX create a userdata object with metatable
    # typedef.LUA_statements={}

    # allow forward declarations to avoid recursive headers
    typedef.forward=typedef.cxx_type


def lookup_c_statements(arg):
    """Look up the c_statements for an argument.
    If the argument type is a template, look for 
    template specific c_statements.
    """
    attrs = arg.attrs
    argtype = arg.typename
    arg_typedef = Typedef.lookup(argtype)

    c_statements = arg_typedef.c_statements
    if 'template' in attrs:
        cxx_T = attrs['template']
        cxx_T = Typedef.resolve_alias(cxx_T)
        c_statements = arg_typedef.c_templates.get(
            cxx_T, c_statements)
        arg_typedef = Typedef.lookup(cxx_T)
    return arg_typedef, c_statements
