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
Generate Python module for C++ code.

Entire library in a single header.
One Extension module per class


Variables prefixes used by generated code:
SH_     C or C++ version of argument
SHPy_   Python object which corresponds to the argument
SHTPy_  A temporary object, usually from PyArg_Parse
        to be converted to SHPy_ object.
SHDPy_  PyArray_Descr object

"""
from __future__ import print_function
from __future__ import absolute_import

import collections
import re

from . import declast
from . import typemap
from . import util
from . import todict
from .util import wformat, append_format

# If multiple values are returned, save up into to build a tuple to return.
# else build value from ctor, then return ctorvar.
# The value may be built earlier (bool, array), if so ctor will be None.
# format   - Format arg to PyBuild_Tuple
# vargs    - Variable for PyBuild_Tuple
# ctor     - Code to construct a Python object
# ctorvar  - Variable created by ctor
BuildTuple = collections.namedtuple(
    'BuildTuple', 'format vargs ctor ctorvar')

# map c types to numpy types
c_to_numpy = dict(
#    'NPY_BOOL',
#    'NPY_BYTE',
#    'NPY_UBYTE',
#    'NPY_SHORT',
#    'NPY_USHORT',
    int='NPY_INT',
#    'NPY_UINT',
    long='NPY_LONG',
#    'NPY_ULONG',
#    'NPY_LONGLONG',
#    'NPY_ULONGLONG',
    float='NPY_FLOAT',
    double='NPY_DOUBLE',
#    'NPY_LONGDOUBLE',
#    'NPY_CFLOAT',
#    'NPY_CDOUBLE',
#    'NPY_CLONGDOUBLE',
#    'NPY_OBJECT',
)


class Wrapp(util.WrapperMixin):
    """Generate Python bindings.
    """

    def __init__(self, newlibrary, config, splicers):
        self.newlibrary = newlibrary
        self.language = newlibrary.language
        self.patterns = self.newlibrary.patterns
        self.config = config
        self.log = config.log
        self._init_splicer(splicers)
        self.comment = '//'
        self.cont = ''
        self.linelen = newlibrary.options.C_line_length
        self.need_numpy = False
        self.enum_impl = []

    def XXX_begin_output_file(self):
        """Start a new class for output"""
        pass

    def XXX_end_output_file(self):
        pass

    def XXX_begin_class(self):
        pass

    def reset_file(self):
        self.PyMethodBody = []
        self.PyMethodDef = []
        self.PyGetSetBody = []
        self.PyGetSetDef = []

    def wrap_library(self):
        newlibrary = self.newlibrary
        options = newlibrary.options
        fmt_library = newlibrary.fmtdict

        if self.language == 'c':
            fmt_library.PY_header_filename_suffix = 'h'
            fmt_library.PY_impl_filename_suffix = 'c'
            fmt_library.PY_extern_C_begin = ''
        else:
            fmt_library.PY_header_filename_suffix = 'hpp'
            fmt_library.PY_impl_filename_suffix = 'cpp'
            fmt_library.PY_extern_C_begin = 'extern "C" '

        # Format variables
        newlibrary.eval_template('PY_module_filename')
        newlibrary.eval_template('PY_header_filename')
        newlibrary.eval_template('PY_helper_filename')
        fmt_library.PY_obj = 'obj'   # name of cpp class pointer in PyObject
        fmt_library.PY_PyObject = 'PyObject'
        fmt_library.PY_param_self = 'self'
        fmt_library.PY_param_args = 'args'
        fmt_library.PY_param_kwds = 'kwds'
        fmt_library.PY_used_param_self = False
        fmt_library.PY_used_param_args = False
        fmt_library.PY_used_param_kwds = False
        fmt_library.PY_this_call = fmt_library.namespace_scope

        # Variables to accumulate output lines
        self.py_type_object_creation = []
        self.py_type_extern = []
        self.py_type_structs = []
        self.py_helper_definition = []
        self.py_helper_declaration = []
        self.py_helper_prototypes = []
        self.py_helper_functions = []

        # preprocess all classes first to allow them to reference each other
        for node in newlibrary.classes:
            if not node.options.wrap_python:
                continue
            typedef = node.typedef
            fmt = node.fmtdict
            typedef.PY_format = 'O'

            # PyTypeObject for class
            node.eval_template('PY_PyTypeObject')

            # PyObject for class
            node.eval_template('PY_PyObject')

            fmt.PY_to_object_func = wformat(
                'PP_{cxx_class}_to_Object', fmt)
            fmt.PY_from_object_func = wformat(
                'PP_{cxx_class}_from_Object', fmt)

            typedef.PY_PyTypeObject = fmt.PY_PyTypeObject
            typedef.PY_PyObject = fmt.PY_PyObject
            typedef.PY_to_object = fmt.PY_to_object_func
            typedef.PY_from_object = fmt.PY_from_object_func

        self._push_splicer('class')
        for node in newlibrary.classes:
            if not node.options.wrap_python:
                continue
            name = node.name
            self.reset_file()
            self._push_splicer(name)
            self.wrap_class(node)
            self.write_extension_type(newlibrary, node)
            self._pop_splicer(name)
        self._pop_splicer('class')

        self.reset_file()
        self.wrap_enums(None)

        if newlibrary.functions:
            self._push_splicer('function')
#            self._begin_class()
            self.wrap_functions(None, newlibrary.functions)
            self._pop_splicer('function')

        self.write_header(newlibrary)
        self.write_module(newlibrary)
        self.write_helper()

    def wrap_enums(self, cls):
        """Wrap enums for library or cls
        """
        if cls is None:
            enums = self.newlibrary.enums
        else:
            enums = cls.enums
        if not enums:
            return
        self._push_splicer('enums')
        for enum in enums:
            self.wrap_enum(enum, cls)
        self._pop_splicer('enums')

    def wrap_enum(self, node, cls):
        """Wrap an enumeration.
        If module, use PyModule_AddIntConstant.
        If class, create a descriptor.
        Without a setter, it will be read-only.
        """
        options = node.options
        fmt_enum = node.fmtdict
        fmtmembers = node._fmtmembers

        ast = node.ast
        output = self.enum_impl
        if cls is None:
            # library enumerations
            # m is module pointer from module_middle
            output.append('')
            append_format(output, '// enumeration {enum_name}', node.fmtdict)
            for member in ast.members:
                fmt_id = fmtmembers[member.name]
                append_format(output,
                              'PyModule_AddIntConstant(m, "{enum_member_name}",'
                              ' {namespace_scope}{enum_member_name});', fmt_id)
        else:
            output.append('{+')
            append_format(output, '// enumeration {enum_name}', node.fmtdict)
            output.append('PyObject *tmp_value;')
            for member in ast.members:
                fmt_id = fmtmembers[member.name]
                append_format(output,
                              'tmp_value = PyLong_FromLong('
                              '{namespace_scope}{enum_member_name});\n'
                              'PyDict_SetItemString('
                              '(PyObject*) {PY_PyTypeObject}.tp_dict,'
                              ' "{enum_member_name}", tmp_value);\n'
                              'Py_DECREF(tmp_value);', fmt_id)
            output.append('-}')


    def wrap_class(self, node):
        self.log.write("class {1.name}\n".format(self, node))
        name = node.name
        unname = util.un_camel(name)
        typedef = typemap.Typedef.lookup(name)

        options = node.options
        fmt_class = node.fmtdict

        node.eval_template('PY_type_filename')
        fmt_class.PY_this_call = wformat('self->{PY_obj}->', fmt_class)

        self.create_class_helper_functions(node)

        self.py_type_object_creation.append(wformat("""
// {cxx_class}
{PY_PyTypeObject}.tp_new   = PyType_GenericNew;
{PY_PyTypeObject}.tp_alloc = PyType_GenericAlloc;
if (PyType_Ready(&{PY_PyTypeObject}) < 0)
+return RETVAL;-
Py_INCREF(&{PY_PyTypeObject});
PyModule_AddObject(m, "{cxx_class}", (PyObject *)&{PY_PyTypeObject});
""", fmt_class))
        self.py_type_extern.append(wformat(
            'extern PyTypeObject {PY_PyTypeObject};', fmt_class))

        self._create_splicer('C_declaration', self.py_type_structs)
        self.py_type_structs.append('')
        self.py_type_structs.append('typedef struct {')
        self.py_type_structs.append('PyObject_HEAD')
        self.py_type_structs.append(1)
        append_format(self.py_type_structs, '{namespace_scope}{cxx_class} * {PY_obj};', fmt_class)
        self._create_splicer('C_object', self.py_type_structs)
        self.py_type_structs.append(-1)
        self.py_type_structs.append(wformat('}} {PY_PyObject};', fmt_class))

        self.wrap_enums(node)

        # wrap methods
        self.tp_init_default = '0'
        self._push_splicer('method')
        self.wrap_functions(node, node.functions)
        self._pop_splicer('method')

    def create_class_helper_functions(self, node):
        """Create some helper functions to and from a PyObject.
        These functions are used by PyArg_ParseTupleAndKeywords
        and Py_BuildValue node is a C++ class.
        """
        fmt = node.fmtdict

        fmt.PY_capsule_name = wformat('PY_{cxx_class}_capsule_name', fmt)

        self._push_splicer('helper')
        append_format(self.py_helper_definition,
                      'const char *{PY_capsule_name} = "{cxx_class}";', fmt)
        append_format(self.py_helper_declaration,
                      'extern const char *{PY_capsule_name};', fmt)

        # To
        to_object = wformat("""PyObject *voidobj;
PyObject *args;
PyObject *rv;

voidobj = PyCapsule_New(addr, {PY_capsule_name}, NULL);
args = PyTuple_New(1);
PyTuple_SET_ITEM(args, 0, voidobj);
rv = PyObject_Call((PyObject *) &{PY_PyTypeObject}, args, NULL);
Py_DECREF(args);
return rv;""", fmt)
        to_object = to_object.split('\n')

        proto = wformat(
            'PyObject *{PY_to_object_func}({namespace_scope}{cxx_class} *addr)', fmt)
        self.py_helper_prototypes.append(proto + ';')

        self.py_helper_functions.append('')
        self.py_helper_functions.append(proto)
        self.py_helper_functions.append('{')
        self.py_helper_functions.append(1)
        self._create_splicer('to_object', self.py_helper_functions, to_object)
        self.py_helper_functions.append(-1)
        self.py_helper_functions.append('}')

        # From
        from_object = wformat("""if (obj->ob_type != &{PY_PyTypeObject}) {{
    // raise exception
    return 0;
}}
{PY_PyObject} * self = ({PY_PyObject} *) obj;
*addr = self->{PY_obj};
return 1;""", fmt)
        from_object = from_object.split('\n')

        proto = wformat(
            'int {PY_from_object_func}(PyObject *obj, void **addr)', fmt)
        self.py_helper_prototypes.append(proto + ';')

        self.py_helper_functions.append('')
        self.py_helper_functions.append(proto)
        self.py_helper_functions.append('{')
        self.py_helper_functions.append(1)
        self._create_splicer(
            'from_object', self.py_helper_functions, from_object)
        self.py_helper_functions.append(-1)
        self.py_helper_functions.append('}')

        self._pop_splicer('helper')

    def allocatable_blk(self, allocatable, node, arg, fmt_arg):
        """Allocate NumPy Array.
        Assumes intent(out)
        """
        self.need_numpy = True
        fmt_arg.py_type = 'PyObject'
        fmt_arg.numpy_type = c_to_numpy[fmt_arg.c_type]

        allocargs, descr_code = attr_allocatable(self.language, allocatable, node, arg)

        asgn = ('{py_var} = %s;' %
                do_cast(
                    self.language, 'reinterpret', 'PyArrayObject *',
                    'PyArray_NewLikeArray(\t%s,\t %s,\t %s,\t %s)' % allocargs))
        if self.language == 'c++':
            cast = '{cxx_decl} = %s;' % do_cast(
                self.language, 'static', '{cxx_type} *', 'PyArray_DATA({py_var})')
        else:
            # No cast needed for void * in C
            cast = '{cxx_decl} = PyArray_DATA({py_var});'

        blk = dict(
#             cxx_local_var = 'pointer',
            goto_fail = True,
            decl = [
                'PyArrayObject * {py_var} = NULL;',
            ],
            pre_call  = [
                descr_code + asgn,
                'if ({py_var} == NULL)', '+goto fail;-',
                cast,
            ],
            post_call = None,   # Object already created
#            cleanup = [
#                'Py_DECREF({pytmp_var});'
#            ],
            fail = [
                'Py_XDECREF({py_var});'
            ],
        )
        return blk

    def dimension_blk(self, arg, fmt_arg):
        """Create code needed for a dimensioned array.
        Convert it to use Numpy.
        """
        self.need_numpy = True
        fmt_arg.pytmp_var = 'SHTPy_' + fmt_arg.c_var
        fmt_arg.py_type = 'PyObject'
        fmt_arg.numpy_type = c_to_numpy[fmt_arg.c_type]
        intent = arg.attrs['intent']
        if intent == 'in':
            fmt_arg.numpy_intent = 'NPY_ARRAY_IN_ARRAY'
        else:
            fmt_arg.numpy_intent = 'NPY_ARRAY_INOUT_ARRAY'

        asgn = ('{py_var} = %s;' %
                do_cast(self.language, 'reinterpret', 'PyArrayObject *', 'PyArray_FROM_OTF('
                        '\t{pytmp_var},\t {numpy_type},\t {numpy_intent})'))

        if self.language == 'c++':
            cast = '{cxx_decl} = %s;' % do_cast(
                self.language, 'static', '{cxx_type} *', 'PyArray_DATA({py_var})')
        else:
            # No cast needed for void * in C
            cast = '{cxx_decl} = PyArray_DATA({py_var});'

        blk = dict(
            # Declare variables here that are used by parse or referenced in fail.
            goto_fail = True,
            decl = [
                '{py_type} * {pytmp_var};',
                'PyArrayObject * {py_var} = NULL;',
            ],
            post_parse = [
                asgn,
                'if ({py_var} == NULL) {{+',
                'PyErr_SetString(PyExc_ValueError,'
                '\t "{c_var} must be a 1-D array of {c_type}");',
                'goto fail;',
                '-}}',
            ],
            pre_call  = [
                cast,
            ],
        )

        if intent == 'in':
            blk['cleanup'] = ['Py_DECREF({py_var});']
            blk['fail'] = ['Py_XDECREF({py_var});']
        else:
            blk['post_call'] = None   # Object already created

        return blk

    def implied_blk(self, node, arg, pre_call):
        """Add the implied attribute to the pre_call block.

        Called after all input arguments have their fmtpy dictionary
        updated.
        Added into wrapper after post_parse code is inserted --
        i.e. all intent in,inout arguments have been evaluated
        and PyArrayObjects created.
        """
        implied = arg.attrs.get('implied', None)
        if implied:
            fmt = node._fmtargs[arg.name]['fmtpy']
            fmt.pre_call_intent = py_implied(implied, node)
            append_format(pre_call, '{cxx_decl} = {pre_call_intent};', fmt)

    def intent_out(self, typedef, intent_blk, fmt, post_call):
        """Add code for post-call.
        Create PyObject from C++ value to return.

        typedef - typedef of C++ variable.
        fmt - format dictionary
        post_call   - always called to construct objects

        Return a BuildTuple instance.
        """

        fmt.PyObject = typedef.PY_PyObject or 'PyObject'
        fmt.PyTypeObject = typedef.PY_PyTypeObject

        if 'post_call' in intent_blk:
            cmd_list = intent_blk['post_call']
            # If post_call is None, the Object has already been created
            if cmd_list is not None:
                for cmd in cmd_list:
                    append_format(post_call, cmd, fmt)
            format = 'O'
            vargs = fmt.py_var
            ctor = None
            ctorvar = fmt.py_var
        else:
            # Decide values for Py_BuildValue
            format = typedef.PY_format
            vargs = typedef.PY_build_arg
            if not vargs:
                vargs = '{cxx_var}'
            vargs = wformat(vargs, fmt)

            if typedef.PY_ctor:
                ctor = wformat('{PyObject} * {py_var} = ' + typedef.PY_ctor
                           + ';', fmt)
                ctorvar = fmt.py_var
            else:
                fmt.PY_format = format
                fmt.vargs = vargs
                ctor = wformat(
                    '{PyObject} * {py_var} = '
                    'Py_BuildValue("{PY_format}", {vargs});', fmt)
                ctorvar = fmt.py_var
                
        return BuildTuple(format, vargs, ctor, ctorvar)

    def wrap_functions(self, cls, functions):
        """Wrap functions for a library or class.
        Compute overloading map.
        cls - C++ class
        """
        overloaded_methods = {}
        for function in functions:
            flist = overloaded_methods. \
                setdefault(function.ast.name, [])
            if not function._cxx_overload:
                continue
            if not function.options.wrap_python:
                continue
            flist.append(function)
        self.overloaded_methods = overloaded_methods

        for function in functions:
            self.wrap_function(cls, function)

        self.multi_dispatch(functions)

    def wrap_function(self, cls, node):
        """Write a Python wrapper for a C++ function.

        cls  - class node or None for functions
        node - function/method node

        fmt.c_var   - name of variable in PyArg_ParseTupleAndKeywords
        fmt.cxx_var - name of variable in c++ call.
        fmt.py_var  - name of PyObject variable

        # Used to prevent compiler warnings about unused variables.
        fmt.PY_used_param_args - True/False if parameter args is used
        fmt.PY_used_param_kwds - True/False if parameter kwds is used

        fmt.PY_error_return - 'NULL' for all wrappers except constructors
                              which are called via tp_init and require -1.
        """
        options = node.options
        if not options.wrap_python:
            return

        if cls:
            cls_function = 'method'
        else:
            cls_function = 'function'
        self.log.write("Python {0} {1.declgen}\n".format(cls_function, node))

        fmt_func = node.fmtdict
        fmtargs = node._fmtargs
        fmt = util.Scope(fmt_func)
        fmt.PY_doc_string = 'documentation'

        ast = node.ast
        CXX_subprogram = ast.get_subprogram()
        result_type = ast.typename
        is_ctor = ast.fattrs.get('_constructor', False)
        is_dtor = ast.fattrs.get('_destructor', False)
#        is_const = ast.const
        ml_flags = []

        if cls:
            fmt.PY_used_param_self = True
            if 'static' in ast.storage:
                ml_flags.append('METH_STATIC')
                fmt_func.PY_this_call = fmt_func.namespace_scope + fmt_func.class_scope

        if is_dtor:
            # Added in tp_del from write_tp_func.
            return
        elif is_ctor:
            fmt_func.PY_type_method = 'tp_init'
            node.eval_template('PY_type_impl')
            fmt_func.PY_name_impl = fmt_func.PY_type_impl
            self.tp_init_default = fmt_func.PY_type_impl
            fmt.PY_error_return = '-1'
        else:
            node.eval_template('PY_name_impl')
            fmt.PY_error_return = 'NULL'

        if node.return_this:
            result_type = 'void'
            CXX_subprogram = 'subroutine'

        result_typedef = typemap.Typedef.lookup(result_type)

        # XXX if a class, then knock off const since the PyObject
        # is not const, otherwise, use const from result.
# This has been replaced by gen_arg methods, but not sure about const.
#        if result_typedef.base == 'shadow':
#            is_const = False
#        else:
#            is_const = None
        if CXX_subprogram == 'function':
            fmt_result0 = node._fmtresult
            fmt_result = fmt_result0.setdefault('fmtpy', util.Scope(fmt)) # fmt_func
            if result_typedef.cxx_to_c is None:
                fmt_result.cxx_var = wformat('{C_local}{C_result}', fmt_result)
            else:
                fmt_result.cxx_var = wformat('{CXX_local}{C_result}', fmt_result)
            CXX_result = ast
            fmt.C_rv_decl = CXX_result.gen_arg_as_cxx(
                name=fmt_result.cxx_var, params=None, continuation=True)
            if CXX_result.is_pointer():
                fmt_result.cxx_addr = ''
                fmt_result.cxx_deref = '->'
            else:
                fmt_result.cxx_addr = '&'
                fmt_result.cxx_deref = '.'
            fmt_result.c_var = fmt_result.cxx_var
            fmt_result.py_var = fmt.PY_result
#            fmt_pattern = fmt_result

        PY_code = []

        # arguments to PyArg_ParseTupleAndKeywords
        parse_format = []
        parse_vargs = []

        # arguments to Py_BuildValue
        build_tuples = []

        # Code blocks
        PY_decl = []      # variables for function
        post_parse = []
        pre_call = []
        post_call = []    # Create objects passed to PyBuildValue
        cleanup_code = []
        fail_code = []

        cxx_call_list = []

        # parse arguments
        # call function based on number of default arguments provided
        default_calls = []   # each possible default call
        found_default = False
        if node._has_default_arg:
            PY_decl.append('Py_ssize_t SH_nargs = 0;')
            PY_code.extend([
                    'if (args != NULL) SH_nargs += PyTuple_Size(args);',
                    'if (kwds != NULL) SH_nargs += PyDict_Size(args);',
                    ])

        goto_fail = False
        args = ast.params
        arg_names = []
        arg_offsets = []
        arg_implied = []  # Collect implied arguments
        offset = 0
        for arg in args:
            arg_name = arg.name
            fmt_arg0 = fmtargs.setdefault(arg_name, {})
            fmt_arg = fmt_arg0.setdefault('fmtpy', util.Scope(fmt))
            fmt_arg.c_var = arg_name
            fmt_arg.cxx_var = arg_name
            fmt_arg.py_var = 'SHPy_' + arg_name

            arg_typedef = typemap.Typedef.lookup(arg.typename)
            # Add formats used by py_statements
            fmt_arg.c_type = arg_typedef.c_type
            fmt_arg.cxx_type = arg_typedef.cxx_type
            if arg.const:
                fmt_arg.c_const = 'const '
            else:
                fmt_arg.c_const = ''
            if arg.is_pointer():
                fmt_arg.c_ptr = ' *'
                fmt_arg.cxx_addr = ''
                fmt_arg.cxx_deref = '->'
            else:
                fmt_arg.c_ptr = ''
                fmt_arg.cxx_addr = '&'
                fmt_arg.cxx_deref = '.'
            attrs = arg.attrs

            dimension = arg.attrs.get('dimension', False)
            pass_var = fmt_arg.c_var  # The variable to pass to the function
            # local_var - 'funcptr', 'pointer', or 'scalar'
            if arg.is_function_pointer():
                fmt_arg.c_decl = arg.gen_arg_as_c(continuation=True)
                fmt_arg.cxx_decl = arg.gen_arg_as_cxx(continuation=True)
                # not sure how function pointers work with Python.
                local_var = 'funcptr'
            elif arg_typedef.base == 'string':
                fmt_arg.c_decl = wformat('{c_const}char * {c_var}', fmt_arg)
#                fmt_arg.cxx_decl = wformat('{c_const}char * {cxx_var}', fmt_arg)
                fmt_arg.cxx_decl = arg.gen_arg_as_cxx()
                local_var = 'pointer'
            elif arg.attrs.get('allocatable', False):
                fmt_arg.c_decl = wformat('{c_type} * {c_var}', fmt_arg)
                fmt_arg.cxx_decl = wformat('{cxx_type} * {cxx_var}', fmt_arg)
                local_var = 'pointer'
            elif arg.attrs.get('dimension', False):
                fmt_arg.c_decl = wformat('{c_type} * {c_var}', fmt_arg)
                fmt_arg.cxx_decl = wformat('{cxx_type} * {cxx_var}', fmt_arg)
                local_var = 'pointer'
            else:
                # non-strings should be scalars
                fmt_arg.c_decl = wformat('{c_type} {c_var}', fmt_arg)
                fmt_arg.cxx_decl = wformat('{cxx_type} {cxx_var}', fmt_arg)
                local_var = 'scalar'

            allocatable = attrs.get('allocatable', False)
            implied = attrs.get('implied', False)
            intent = attrs['intent']
            if implied:
                arg_implied.append(arg)
                intent_blk = {}
            elif allocatable:
                intent_blk = self.allocatable_blk(allocatable, node, arg, fmt_arg)
            elif dimension:
                intent_blk = self.dimension_blk(arg, fmt_arg)
            else:
                py_statements = arg_typedef.py_statements
                stmts = 'intent_' + intent
                intent_blk = py_statements.get(stmts, {})

            goto_fail = goto_fail or intent_blk.get('goto_fail', False)
            cxx_local_var = intent_blk.get('cxx_local_var', '')
            if cxx_local_var:
                # With PY_PyTypeObject, there is no c_var, only cxx_var
                if not arg_typedef.PY_PyTypeObject:
                    fmt_arg.cxx_var = 'SH_' + fmt_arg.c_var
                local_var = cxx_local_var
                pass_var = fmt_arg.cxx_var
                # cxx_deref used with typedef fields like PY_ctor.
                if cxx_local_var == 'scalar':
                    fmt_arg.cxx_deref = '.'
                elif cxx_local_var == 'pointer':
                    fmt_arg.cxx_deref = '->'

            if implied:
                pass
            elif intent in ['inout', 'in']:
                # names to PyArg_ParseTupleAndKeywords
                arg_names.append(arg_name)
                arg_offsets.append('(char *) SH_kwcpp+%d' % offset)
                offset += len(arg_name) + 1

                # XXX default should be handled differently
                if arg.init is not None:
                    if not found_default:
                        parse_format.append('|')  # add once
                        found_default = True
                    # call for default arguments  (num args, arg string)
                    default_calls.append(
                        (len(cxx_call_list), len(post_parse), len(pre_call),
                         ',\t '.join(cxx_call_list)))

                # Declare C variable - may be PyObject.
                # add argument to call to PyArg_ParseTypleAndKeywords
                if dimension:
                    # Use NumPy with dimensioned arguments
                    pass_var = fmt_arg.cxx_var
                    parse_format.append('O')
                    parse_vargs.append('&' + fmt_arg.pytmp_var)
                elif arg_typedef.PY_PyTypeObject:
                    # Expect object of given type
                    # cxx_var is declared by py_statements.intent_out.post_parse.
                    fmt_arg.py_type = arg_typedef.PY_PyObject or 'PyObject'
                    append_format(PY_decl, '{py_type} * {py_var};', fmt_arg)
                    pass_var = fmt_arg.cxx_var
                    parse_format.append(arg_typedef.PY_format)
                    parse_format.append('!')
                    parse_vargs.append('&' + arg_typedef.PY_PyTypeObject)
                    parse_vargs.append('&' + fmt_arg.py_var)
                elif arg_typedef.PY_from_object:
                    # Use function to convert object
                    # cxx_var created directly (no c_var)
                    append_format(PY_decl, '{cxx_decl};', fmt_arg)
                    pass_var = fmt_arg.cxx_var
                    parse_format.append(arg_typedef.PY_format)
                    parse_format.append('&')
                    parse_vargs.append(arg_typedef.PY_from_object)
                    parse_vargs.append('&' + fmt_arg.cxx_var)
                else:
                    append_format(PY_decl, '{c_decl};', fmt_arg)
                    parse_format.append(arg_typedef.PY_format)
                    parse_vargs.append('&' + fmt_arg.c_var)

            if intent in ['inout', 'out']:
                if intent == 'out':
                    if allocatable:
                        pass
                    elif not cxx_local_var:
                        pass_var = fmt_arg.cxx_var
                        append_format(pre_call,
                                      '{cxx_decl};  // intent(out)',
                                      fmt_arg)

                # output variable must be a pointer
                build_tuples.append(self.intent_out(
                    arg_typedef, intent_blk, fmt_arg, post_call))

            # Code to convert parsed values (C or Python) to C++.
            cmd_list = intent_blk.get('decl', [])
            for cmd in cmd_list:
                append_format(PY_decl, cmd, fmt_arg)
            cmd_list = intent_blk.get('post_parse', [])
            for cmd in cmd_list:
                append_format(post_parse, cmd, fmt_arg)
            cmd_list = intent_blk.get('pre_call', [])
            for cmd in cmd_list:
                append_format(pre_call, cmd, fmt_arg)
            cmd_list = intent_blk.get('cleanup', [])
            for cmd in cmd_list:
                append_format(cleanup_code, cmd, fmt_arg)
            cmd_list = intent_blk.get('fail', [])
            for cmd in cmd_list:
                append_format(fail_code, cmd, fmt_arg)

            if intent != 'out' and not cxx_local_var and arg_typedef.c_to_cxx:
                # Make intermediate C++ variable
                # Needed to pass address of variable
                # Helpful with debugging.
                fmt_arg.cxx_var = 'SH_' + fmt_arg.c_var
                fmt_arg.cxx_decl = arg.gen_arg_as_cxx(
                    name=fmt_arg.cxx_var, params=None, continuation=True)
                fmt_arg.cxx_val = wformat(arg_typedef.c_to_cxx, fmt_arg)
                append_format(post_parse, '{cxx_decl} = {cxx_val};', fmt_arg)
                pass_var = fmt_arg.cxx_var

            # Pass correct value to wrapped function.
            if local_var == 'scalar':
                if arg.is_pointer():
                    cxx_call_list.append('&' + pass_var)
                else:
                    cxx_call_list.append(pass_var)
            elif local_var == 'pointer':
                if arg.is_pointer():
                    cxx_call_list.append(pass_var)
                else:
                    cxx_call_list.append('*' + pass_var)
            elif local_var == 'funcptr':
                cxx_call_list.append(pass_var)
            else:
                raise RuntimeError("unexpected value of local_var")

        # Add implied argument initialization to pre_call code
        for arg in arg_implied:
            intent_blk = self.implied_blk(node, arg, pre_call)

        need_blank = False  # needed before next debug header
        if not arg_names:
            # no input arguments
            ml_flags.append('METH_NOARGS')
        else:
            ml_flags.append('METH_VARARGS')
            ml_flags.append('METH_KEYWORDS')
            fmt.PY_used_param_args = True
            fmt.PY_used_param_kwds = True
            need_blank = True

            if self.language == 'c++':
                kw_const = 'const '
                fmt.PyArg_kwlist = 'const_cast<char **>(SHT_kwlist)'
            else:
                kw_const = ''
                fmt.PyArg_kwlist = 'SHT_kwlist'
            PY_decl.append(
                kw_const + 'char *SHT_kwlist[] = {\f"' +
                '",\f"'.join(arg_names)
                + '",\fNULL };')
            parse_format.extend([':', fmt.function_name])
            fmt.PyArg_format = ''.join(parse_format)
            fmt.PyArg_vargs = ',\t '.join(parse_vargs)
            PY_code.append(
                wformat(
                    'if (!PyArg_ParseTupleAndKeywords'
                    '({PY_param_args}, {PY_param_kwds},\t '
                    '"{PyArg_format}",\t {PyArg_kwlist}, '
                    '\t{PyArg_vargs}))', fmt))
            append_format(PY_code, '+return {PY_error_return};-', fmt)

        # call with all arguments
        default_calls.append(
            (len(cxx_call_list), len(post_parse), len(pre_call),
             ',\t '.join(cxx_call_list)))

        # If multiple calls, declare return value once
        # Else delare on call line.
        if found_default:
            if CXX_subprogram == 'function':
                fmt.PY_rv_asgn = fmt_result.cxx_var + ' = '
            PY_code.append('switch (SH_nargs) {')
        else:
            if CXX_subprogram == 'function':
                fmt.PY_rv_asgn = fmt.C_rv_decl + ' = '
        need_rv = False

        # build up code for a function
        for nargs, len_post_parse, len_pre_call, call_list in default_calls:
            if found_default:
                PY_code.append('case %d:' % nargs)
                PY_code.append(1)
                need_blank = False
                if len_post_parse or len_pre_call:
                    # Only add scope if necessary
                    PY_code.append('{')
                    PY_code.append(1)
                    extra_scope = True
                else:
                    extra_scope = False

            if len_post_parse:
                if options.debug:
                    if need_blank:
                        PY_code.append('')
                    PY_code.append('// post_parse')
                PY_code.extend(post_parse[:len_post_parse])
                need_blank = True

            if self.language == 'c++' and goto_fail:
                # Need an extra scope to deal with C++ error
                # error: jump to label 'fail' crosses initialization of ...
                PY_code.append('{')
                PY_code.append(1)
                fail_scope = True
                need_blank = False
            else:
                fail_scope = False
            
            if len_pre_call:
                if options.debug:
                    if need_blank:
                        PY_code.append('')
                    PY_code.append('// pre_call')
                PY_code.extend(pre_call[:len_pre_call])
                need_blank = True
            fmt.PY_call_list = call_list

            if options.debug and need_blank:
                PY_code.append('')

            if is_ctor:
                append_format(PY_code, 'self->{PY_obj} = new {namespace_scope}'
                              '{cxx_class}({PY_call_list});', fmt)
            elif CXX_subprogram == 'subroutine':
                line = wformat(
                    '{PY_this_call}{function_name}({PY_call_list});', fmt)
                PY_code.append(line)
            else:
                need_rv = True
                line = wformat(
                    '{PY_rv_asgn}{PY_this_call}{function_name}({PY_call_list});',
                    fmt)
                PY_code.append(line)

            if node.PY_error_pattern:
                lfmt = util.Scope(fmt)
                lfmt.c_var = fmt.C_result
                lfmt.cxx_var = fmt.C_result
                append_format(PY_code,
                              self.patterns[node.PY_error_pattern], lfmt)

            if found_default:
                PY_code.append('break;')
                PY_code.append(-1)
                if extra_scope:
                    PY_code.append('}')
                    PY_code.append(-1)
                    need_blank = False
        if found_default:
            # PY_code.append('default:')
            # PY_code.append(1)
            # PY_code.append('continue;')  # XXX raise internal error
            # PY_code.append(-1)
            PY_code.append('}')
        else:
            need_rv = False

        if need_rv:
            PY_decl.append(fmt.C_rv_decl + ';')
        if len(PY_decl):
            PY_decl.append('')

        # Compute return value
        if CXX_subprogram == 'function':
            # XXX - wrapc uses result instead of intent_out
            result_blk = result_typedef.py_statements.get('intent_out', {})
            ttt = self.intent_out(result_typedef, result_blk,
                                  fmt_result, post_call)
            # Add result to front of result tuple
            build_tuples.insert(0, ttt)

        # If only one return value, return the ctor
        # else create a tuple with Py_BuildValue.
        if is_ctor:
            return_code = 'return 0;'
        elif not build_tuples:
            return_code = 'Py_RETURN_NONE;'
        elif len(build_tuples) == 1:
            # return a single object already created in build_stmts
            ctor = build_tuples[0].ctor
            if ctor:
                post_call.append(ctor)
            fmt.py_var = build_tuples[0].ctorvar
            return_code = wformat('return (PyObject *) {py_var};', fmt)
        else:
            # create tuple object
            fmt.PyBuild_format = ''.join([ttt.format for ttt in build_tuples])
            fmt.PyBuild_vargs = ',\t '.join([ttt.vargs for ttt in build_tuples])
            append_format(
                post_call, 'PyObject * {PY_result} = '
                'Py_BuildValue("{PyBuild_format}",\t {PyBuild_vargs});',
                fmt)
            return_code = wformat('return {PY_result};', fmt)

        need_blank = False  # put return right after call
        if post_call and not is_ctor:
            # ctor does not need to build return values
            if options.debug:
                PY_code.append('')
                PY_code.append('// post_call')
            PY_code.extend(post_call)
            need_blank = True

        if cleanup_code:
            if options.debug:
                PY_code.append('')
                PY_code.append('// cleanup')
            PY_code.extend(cleanup_code)
            need_blank = True

        if options.debug and need_blank:
            PY_code.append('')
        PY_code.append(return_code)

        if fail_scope:
            PY_code.append(-1)
            PY_code.append('}')
        if goto_fail:
            PY_code.extend(['', '0fail:'])
            PY_code.extend(fail_code)
            append_format(PY_code, 'return {PY_error_return};', fmt)

        PY_impl = [1] + PY_decl + PY_code + [-1]

        expose = True
        if is_ctor:
            expose = False
        if len(self.overloaded_methods[ast.name]) > 1:
            # Only expose a multi-dispatch name, not each overload
            expose = False
        elif found_default:
            # Only one wrapper to deal with default arugments.
            # [C creates a wrapper per default argument]
            fmt = util.Scope(fmt)
            fmt.function_suffix = ''

        fmt.PY_ml_flags = '|'.join(ml_flags)
        self.create_method(node, expose, is_ctor, fmt, PY_impl)

    def create_method(self, node, expose, is_ctor, fmt, PY_impl):
        """Format the function.
        node    = function node to wrap
        expose  = True if expose to user
        is_ctor = True if this is a constructor
        fmt     = dictionary of format values
        PY_impl = list of implementation lines
        """
        body = self.PyMethodBody
        if expose:
            body.extend([
                    '',
                    wformat('static char {PY_name_impl}__doc__[] =', fmt),
                    '"%s"' % fmt.PY_doc_string,
                    ';',
                    ])

        body.append('')
        if is_ctor:
            body.append('static int')
        else:
            body.append('static PyObject *')
        body.append(wformat('{PY_name_impl}(', fmt))

        if fmt.PY_used_param_self:
            body.append(wformat(
                '  {PY_PyObject} *{PY_param_self},', fmt))
        else:
            body.append(wformat(
                '  PyObject *SHROUD_UNUSED({PY_param_self}),', fmt))
        if fmt.PY_used_param_args:
            body.append(wformat(
                '  PyObject *{PY_param_args},', fmt))
        else:
            body.append(wformat(
                '  PyObject *SHROUD_UNUSED({PY_param_args}),', fmt)),
        if fmt.PY_used_param_args:
            body.append(wformat(
                '  PyObject *{PY_param_kwds})', fmt))
        else:
            body.append(wformat(
                '  PyObject *SHROUD_UNUSED({PY_param_kwds}))', fmt))

        body.append('{')
# use function_suffix in splicer name since a single C++ function may
# produce several methods.
# XXX - make splicer name customizable?
#        self._create_splicer(fmt.function_name, self.PyMethodBody, PY_impl)
        if node and node.options.debug:
            self.PyMethodBody.append('// ' + node.declgen)
        self._create_splicer(fmt.underscore_name + fmt.function_suffix,
                             self.PyMethodBody, PY_impl)
        self.PyMethodBody.append('}')

        if expose is True:
            # default name
            self.PyMethodDef.append(
                wformat('{{"{function_name}{function_suffix}",\t '
                        '(PyCFunction){PY_name_impl},\t '
                        '{PY_ml_flags},\t '
                        '{PY_name_impl}__doc__}},', fmt))
#        elif expose is not False:
#            # override name
#            fmt = util.Scope(fmt)
#            fmt.expose = expose
#            self.PyMethodDef.append( wformat('{{"{expose}", (PyCFunction){PY_name_impl}, {PY_ml_flags}, {PY_name_impl}__doc__}},', fmt))

    def write_tp_func(self, node, fmt_type, output):
        """Create functions for tp_init et.al.

        fmt_type - dictionary used with PyTypeObject_template
                   to fill in type function names.
        output - list for generated functions.

        python:
          type: [ repr, str ]

        """
        # Type bodies must be filled in by user, no real way to guess
        # how to implement.
        # Except tp_init (constructor) and tp_del (destructor).
        fmt_func = node.fmtdict
        fmt = util.Scope(fmt_func)
        template = node.options.PY_type_impl_template
        PyObj = fmt_func.PY_PyObject
        if 'type' in node.python:
            selected = node.python['type'][:]
            for auto in [ 'del']:
                # Make some methods are there
                if auto not in selected:
                    selected.append(auto)
        else:
            selected = [ 'del' ]
            
        # Dictionary of methods for bodies
        default_body = dict(
            richcompare=self.not_implemented
        )
        default_body['del'] = self.tp_del

        self._push_splicer('type')
        for typename in typenames:
            tp_name = 'tp_' + typename
            if typename == 'init':
                # The constructor method is used for tp_init
                fmt_type[tp_name] = self.tp_init_default
                continue
            if typename not in selected:
                fmt_type[tp_name] = fmt_type['nullptr']
                continue
            fmt.PY_type_method = tp_name
            func_name = wformat(template, fmt)
            fmt_type[tp_name] = func_name
            tup = typefuncs[typename]
            output.append('static ' + tup[0])
            output.append(('{name} ' + tup[1])  # object used by tup[1]
                          .format(name=func_name, object=PyObj))
            output.append('{')
            default = default_body.get(typename, self.not_implemented_error)
            default = default(typename, tup[2])

            # format and indent default bodies
            fmted = [1]
            for line in default:
                append_format(fmted, line, fmt_func)
            fmted.append(-1)

            self._create_splicer(typename, output, fmted)
            output.append('}')
        self._pop_splicer('type')

    def write_extension_type(self, library, node):
        fmt = node.fmtdict
        fname = fmt.PY_type_filename

        output = []

        output.append(wformat('#include "{PY_header_filename}"', fmt))
        self._push_splicer('impl')

        # Use headers from class if they exist or else library
        header_impl_include = {}
        if node and node.cxx_header:
            for include in node.cxx_header.split():
                header_impl_include[include] = True
        else:
            for include in library.cxx_header.split():
                header_impl_include[include] = True
        self.write_headers(header_impl_include, output)

        self._create_splicer('include', output)
        output.append(cpp_boilerplate)
        self._create_splicer('C_definition', output)
        self._create_splicer('additional_methods', output)
        self._pop_splicer('impl')

        fmt_type = dict(
            PY_module_name=fmt.PY_module_name,
            PY_PyObject=fmt.PY_PyObject,
            PY_PyTypeObject=fmt.PY_PyTypeObject,
            cxx_class=fmt.cxx_class,
            nullptr=' 0', #'NULL',   # 0 will confuse formatter (thinks no indent)
            )
        self.write_tp_func(node, fmt_type, output)

        output.extend(self.PyMethodBody)

        self._push_splicer('impl')
        self._create_splicer('after_methods', output)
        self._pop_splicer('impl')

        output.extend(self.PyGetSetBody)
        if self.PyGetSetDef:
            fmt_type['tp_getset'] = wformat('{PY_prefix}{cxx_class}_getset', fmt)
            output.append(
                wformat('\nstatic PyGetSetDef {tp_getset}[] = {{+', fmt_type))
            output.extend(self.PyGetSetDef)
            self._create_splicer('PyGetSetDef', output)
            output.append('{NULL}            /* sentinel */')
            output.append('-};')
        else:
            fmt_type['tp_getset'] = fmt_type['nullptr']

        fmt_type['tp_methods'] = wformat('{PY_prefix}{cxx_class}_methods', fmt)
        output.append(
            wformat('static PyMethodDef {tp_methods}[] = {{+', fmt_type))
        output.extend(self.PyMethodDef)
        self._create_splicer('PyMethodDef', output)
        output.append('{NULL,   (PyCFunction)NULL, 0, NULL}'
                      '            /* sentinel */')
        output.append('-};')

        output.append(wformat(PyTypeObject_template, fmt_type))

        self.write_output_file(fname, self.config.python_dir, output)

    def multi_dispatch(self, methods):
        """Look for overloaded methods.
        When found, create a method which will call each of the
        overloaded methods looking for the one which will accept
        the given arguments.
        """
        for method, methods in self.overloaded_methods.items():
            if len(methods) < 2:
                continue  # not overloaded

            node = methods[0]
            fmt_func = node.fmtdict
            fmt = util.Scope(fmt_func)
            fmt.function_suffix = ''
            fmt.PY_doc_string = 'documentation'
            fmt.PY_ml_flags = 'METH_VARARGS|METH_KEYWORDS'
            fmt.PY_used_param_self = True
            fmt.PY_used_param_args = True
            fmt.PY_used_param_kwds = True

            is_ctor = node.ast.fattrs.get('_constructor', False)

            body = []
            body.append(1)
            body.append('Py_ssize_t SHT_nargs = 0;')
            body.extend([
                    'if (args != NULL) SHT_nargs += PyTuple_Size(args);',
                    'if (kwds != NULL) SHT_nargs += PyDict_Size(args);',
                    ])
            if is_ctor:
                fmt.PY_type_method = 'tp_init'
                fmt.PY_name_impl = wformat(
                    node.options.PY_type_impl_template, fmt)
                fmt.PY_type_impl = fmt.PY_name_impl
                self.tp_init_default = fmt.PY_type_impl
                return_code = 'return rv;'
                return_arg = 'rv'
                fmt.PY_error_return = '-1'
                self.tp_init_default = fmt.PY_name_impl
                body.append('int rv;')
                expose = False
            else:
                fmt.PY_name_impl = wformat(
                    node.options.PY_name_impl_template, fmt)
                return_code = 'return rvobj;'
                return_arg = 'rvobj'
                fmt.PY_error_return = 'NULL'
                body.append('PyObject *rvobj;')
                expose = True

            for overload in methods:
                if overload._nargs:
                    body.append('if (SHT_nargs >= %d && SHT_nargs <= %d) {'
                                % overload._nargs)
                else:
                    body.append('if (SHT_nargs == %d) {' %
                                len(overload.ast.params))
                body.append(1)
                append_format(body,
                              return_arg + ' = {PY_name_impl}(self, args, kwds);',
                              overload.fmtdict)
                body.append('if (!PyErr_Occurred()) {')
                body.append(1)
                body.append(return_code)
                body.append(-1)
                body.append('} else if (! PyErr_ExceptionMatches'
                            '(PyExc_TypeError)) {')
                body.append(1)
                body.append(return_code)
                body.append(-1)
                body.append('}')
                body.append('PyErr_Clear();')
                body.append(-1)
                body.append('}')

            body.append('PyErr_SetString(PyExc_TypeError, '
                        '"wrong arguments multi-dispatch");')
            append_format(body, 'return {PY_error_return};', fmt)
            body.append(-1)

            self.create_method(None, expose, is_ctor, fmt, body)

    def write_header(self, node):
        # node is library
        options = node.options
        fmt = node.fmtdict
        fname = fmt.PY_header_filename

        output = []

        # add guard
        guard = fname.replace(".", "_").upper()
        output.extend([
                '#ifndef %s' % guard,
                '#define %s' % guard,
                ])

        output.append('#include <Python.h>')

        self._push_splicer('header')
        self._create_splicer('include', output)

        # forward declare classes for helpers
        blank = True
        for cls in node.classes:
            if cls.options.wrap_python:
                if blank:
                    output.append('')
                    output.append('// forward declare classes')
                    blank = False
                self.namespace(self.newlibrary, cls, 'begin', output)
                output.append('class {};'.format(cls.name))
                self.namespace(self.newlibrary, cls, 'end', output, comment=False)

        if self.py_type_extern:
            output.append('')
            output.extend(self.py_type_extern)
        output.append('')
        self._create_splicer('C_declaration', output)
        self._pop_splicer('header')

        output.append('')
        output.append('// helper functions')
        output.extend(self.py_helper_declaration)
        output.extend(self.py_helper_prototypes)

        output.append('')
        output.extend(self.py_type_structs)
        output.append(wformat("""
extern PyObject *{PY_prefix}error_obj;

#if PY_MAJOR_VERSION >= 3
{PY_extern_C_begin}PyMODINIT_FUNC PyInit_{PY_module_name}(void);
#else
{PY_extern_C_begin}PyMODINIT_FUNC init{PY_module_name}(void);
#endif
""", fmt))
        output.append('#endif  /* %s */' % guard)
        self.write_output_file(fname, self.config.python_dir, output)

    def write_module(self, node):
        # node is library.
        options = node.options
        fmt = node.fmtdict
        fname = fmt.PY_module_filename

        fmt.PY_library_doc = 'library documentation'

        output = []

        output.append(wformat('#include "{PY_header_filename}"', fmt))
        if self.need_numpy:
            output.append('#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION')
            output.append('#include "numpy/arrayobject.h"')
        for include in node.cxx_header.split():
            output.append('#include "%s"' % include)
        output.append('')
        self._create_splicer('include', output)
        output.append(cpp_boilerplate)
        output.append('')
        self._create_splicer('C_definition', output)

        output.append(wformat('PyObject *{PY_prefix}error_obj;', fmt))

        self._create_splicer('additional_functions', output)
        output.extend(self.PyMethodBody)

        output.append(
            wformat('static PyMethodDef {PY_prefix}methods[] = {{', fmt))
        output.extend(
            self.PyMethodDef)
        output.append(
            '{NULL,   (PyCFunction)NULL, 0, NULL}            /* sentinel */')
        output.append(
            '};')

        output.append(wformat(module_begin, fmt))
        self._create_splicer('C_init_locals', output)
        output.append(wformat(module_middle, fmt))
        if self.need_numpy:
            output.append('import_array();')
        output.extend(self.py_type_object_creation)
        output.extend(self.enum_impl)
        output.append(wformat(module_middle2, fmt))
        self._create_splicer('C_init_body', output)
        output.append(wformat(module_end, fmt))

        self.write_output_file(fname, self.config.python_dir, output)

    def write_helper(self):
        node = self.newlibrary
        fmt = node.fmtdict
        output = []
        output.append(wformat('#include "{PY_header_filename}"', fmt))
        output.extend(self.py_helper_definition)
        output.append('')
        output.extend(self.py_helper_functions)
        self.write_output_file(
            fmt.PY_helper_filename, self.config.python_dir, output)

    def not_implemented_error(self, msg, ret):
        '''A standard splicer for unimplemented code
        ret is the return value (NULL or -1 or '')
        '''
        lines = [
            "PyErr_SetString(PyExc_NotImplementedError, \"%s\");" % msg,
            ]
        if ret:
            lines.append("return %s;" % ret)
        else:
            lines.append('return;')
        return lines

    def not_implemented(self, msg, ret):
        '''A standard splicer for rich comparison
        '''
        return [
            'Py_INCREF(Py_NotImplemented);',
            'return Py_NotImplemented;'
            ]

    def tp_del(self, msg, ret):
        """default method for tp_del.
          msg = 'del'
          ret = ''
        """
        return [
            'delete self->{PY_obj};',
            'self->{PY_obj} = NULL;',
        ]

# --- Python boiler plate

# Avoid warning errors about unused parameters
# Include in each source file and not the header file because
# we don't want to pollute the user's files.
cpp_boilerplate = """
#ifdef __cplusplus
#define SHROUD_UNUSED(param)
#else
#define SHROUD_UNUSED(param) param
#endif

#if PY_MAJOR_VERSION >= 3
#define PyInt_FromLong PyLong_FromLong
#define PyString_FromString PyUnicode_FromString
#define PyString_FromStringAndSize PyUnicode_FromStringAndSize
#endif"""

typenames = [
    'dealloc', 'print', 'compare',
    'getattr', 'setattr',  # deprecated
    'getattro', 'setattro',
    'repr', 'hash', 'call', 'str',
    'init', 'alloc', 'new', 'free', 'del',
    'richcompare',
]


# return type, prototype, default return value
typefuncs = {
    'dealloc': (
        'void',
        '({object} *self)',
        ''),
    'print': (
        'int',
        '({object} *self, FILE *fp, int flags)',
        '-1'),
    'getattr': (
        'PyObject *',
        '({object} *self, char *name)',
        'NULL'),
    'setattr': (
        'int',
        '({object} *self, char *name, PyObject *value)',
        '-1'),
    'compare': (
        'int',
        '({object} *self, PyObject *)',
        '-1'),
    'repr': (
        'PyObject *',
        '({object} *self)',
        'NULL'),
    'hash': (
        'long',
        '({object} *self)',
        '-1'),
    'call': (
        'PyObject *',
        '({object} *self, PyObject *args, PyObject *kwds)',
        'NULL'),
    'str': (
        'PyObject *',
        '({object} *self)',
        'NULL'),
    'getattro': (
        'PyObject *',
        '({object} *self, PyObject *name)',
        'NULL'),
    'setattro': (
        'int',
        '({object} *self, PyObject *name, PyObject *value)',
        '-1'),
    'init': (
        'int',
        '({object} *self, PyObject *args, PyObject *kwds)',
        '-1'),
    'alloc': (
        'PyObject *',
        '(PyTypeObject *type, Py_ssize_t nitems)',
        'NULL'),
    'new': (
        'PyObject *',
        '(PyTypeObject *type, PyObject *args, PyObject *kwds)',
        'NULL'),
    'free': (
        'void',
        '(void *op)',
        ''),
    'del': (
        'void',
        '({object} *self)',
        ''),
    'richcompare': (
        'PyObject *',
        '({object} *self, PyObject *other, int opid)',
        ''),
}

### Note that these strings have some format character to control indenting
#  + indent
#  - deindent
#  0 noindention

PyTypeObject_template = """
static char {cxx_class}__doc__[] =
"virtual class"
;

/* static */
PyTypeObject {PY_PyTypeObject} = {{+
PyVarObject_HEAD_INIT(NULL, 0)
"{PY_module_name}.{cxx_class}",                       /* tp_name */
sizeof({PY_PyObject}),         /* tp_basicsize */
{nullptr},                              /* tp_itemsize */
/* Methods to implement standard operations */
(destructor){tp_dealloc},                 /* tp_dealloc */
(printfunc){tp_print},                   /* tp_print */
(getattrfunc){tp_getattr},                 /* tp_getattr */
(setattrfunc){tp_setattr},                 /* tp_setattr */
0#if PY_MAJOR_VERSION >= 3
{nullptr},                               /* tp_reserved */
0#else
(cmpfunc){tp_compare},                     /* tp_compare */
0#endif
(reprfunc){tp_repr},                    /* tp_repr */
/* Method suites for standard classes */
{nullptr},                              /* tp_as_number */
{nullptr},                              /* tp_as_sequence */
{nullptr},                              /* tp_as_mapping */
/* More standard operations (here for binary compatibility) */
(hashfunc){tp_hash},                    /* tp_hash */
(ternaryfunc){tp_call},                 /* tp_call */
(reprfunc){tp_str},                    /* tp_str */
(getattrofunc){tp_getattro},                /* tp_getattro */
(setattrofunc){tp_setattro},                /* tp_setattro */
/* Functions to access object as input/output buffer */
{nullptr},                              /* tp_as_buffer */
/* Flags to define presence of optional/expanded features */
Py_TPFLAGS_DEFAULT,             /* tp_flags */
{cxx_class}__doc__,         /* tp_doc */
/* Assigned meaning in release 2.0 */
/* call function for all accessible objects */
(traverseproc){nullptr},                /* tp_traverse */
/* delete references to contained objects */
(inquiry){nullptr},                     /* tp_clear */
/* Assigned meaning in release 2.1 */
/* rich comparisons */
(richcmpfunc){tp_richcompare},                 /* tp_richcompare */
/* weak reference enabler */
{nullptr},                              /* tp_weaklistoffset */
/* Added in release 2.2 */
/* Iterators */
(getiterfunc){nullptr},                 /* tp_iter */
(iternextfunc){nullptr},                /* tp_iternext */
/* Attribute descriptor and subclassing stuff */
{tp_methods},                             /* tp_methods */
{nullptr},                              /* tp_members */
{tp_getset},                             /* tp_getset */
{nullptr},                              /* tp_base */
{nullptr},                              /* tp_dict */
(descrgetfunc){nullptr},                /* tp_descr_get */
(descrsetfunc){nullptr},                /* tp_descr_set */
{nullptr},                              /* tp_dictoffset */
(initproc){tp_init},                   /* tp_init */
(allocfunc){tp_alloc},                  /* tp_alloc */
(newfunc){tp_new},                    /* tp_new */
(freefunc){tp_free},                   /* tp_free */
(inquiry){nullptr},                     /* tp_is_gc */
{nullptr},                              /* tp_bases */
{nullptr},                              /* tp_mro */
{nullptr},                              /* tp_cache */
{nullptr},                              /* tp_subclasses */
{nullptr},                              /* tp_weaklist */
(destructor){tp_del},                 /* tp_del */
{nullptr},                              /* tp_version_tag */
0#if PY_MAJOR_VERSION >= 3
(destructor){nullptr},                  /* tp_finalize */
0#endif
-}};"""


module_begin = """
/*
 * init{library_lower} - Initialization function for the module
 * *must* be called init{library_lower}
 */
static char {PY_prefix}_doc__[] =
"{PY_library_doc}"
;

struct module_state {{
    PyObject *error;
}};

#if PY_MAJOR_VERSION >= 3
#define GETSTATE(m) ((struct module_state*)PyModule_GetState(m))
#else
#define GETSTATE(m) (&_state)
static struct module_state _state;
#endif

#if PY_MAJOR_VERSION >= 3
static int {library_lower}_traverse(PyObject *m, visitproc visit, void *arg) {{
    Py_VISIT(GETSTATE(m)->error);
    return 0;
}}

static int {library_lower}_clear(PyObject *m) {{
    Py_CLEAR(GETSTATE(m)->error);
    return 0;
}}

static struct PyModuleDef moduledef = {{
    PyModuleDef_HEAD_INIT,
    "{library_lower}", /* m_name */
    {PY_prefix}_doc__, /* m_doc */
    sizeof(struct module_state), /* m_size */
    {PY_prefix}methods, /* m_methods */
    NULL, /* m_reload */
    {library_lower}_traverse, /* m_traverse */
    {library_lower}_clear, /* m_clear */
    NULL  /* m_free */
}};

#define RETVAL m
#define INITERROR return NULL
#else
#define RETVAL
#define INITERROR return
#endif

{PY_extern_C_begin}PyMODINIT_FUNC
#if PY_MAJOR_VERSION >= 3
PyInit_{PY_module_name}(void)
#else
init{PY_module_name}(void)
#endif
{{+
PyObject *m = NULL;
const char * error_name = "{library_lower}.Error";
"""

module_middle = """

/* Create the module and add the functions */
0#if PY_MAJOR_VERSION >= 3
m = PyModule_Create(&moduledef);
0#else
m = Py_InitModule4("{PY_module_name}", {PY_prefix}methods,\t
+{PY_prefix}_doc__,
(PyObject*)NULL,PYTHON_API_VERSION);
0#endif
-if (m == NULL)
+return RETVAL;-
struct module_state *st = GETSTATE(m);
"""

module_middle2 = """
{PY_prefix}error_obj = PyErr_NewException((char *) error_name, NULL, NULL);
if ({PY_prefix}error_obj == NULL)
+return RETVAL;-
st->error = {PY_prefix}error_obj;
PyModule_AddObject(m, "Error", st->error);
"""

module_end = """
/* Check for errors */
if (PyErr_Occurred())
+Py_FatalError("can't initialize module {PY_module_name}");-
return RETVAL;
-}}
"""


class ToImplied(todict.PrintNode):
    """Convert implied expression to Python wrapper code.

    expression has already been checked for errors by generate.check_implied.
    Convert functions:
      size  -  PyArray_SIZE
    """
    def __init__(self, expr, func):
        super(ToImplied, self).__init__()
        self.expr = expr
        self.func = func

    def visit_Identifier(self, node):
        # Look for functions
        if node.args == None:
            return node.name
        elif node.name == 'size':
            # size(arg)
            argname = node.args[0].name
            arg = self.func.ast.find_arg_by_name(argname)
            fmt = self.func._fmtargs[argname]['fmtpy']
            return wformat('PyArray_SIZE({py_var})', fmt)
        else:
            return self.param_list(node)

def py_implied(expr, func):
    """Convert string to Python code.
    """
    node = declast.ExprParser(expr).expression()
    visitor = ToImplied(expr, func)
    return visitor.visit(node)

def attr_allocatable(language, allocatable, node, arg):
    """parse allocatable and return tuple of 
      (prototype, order, descr, subok)

    Valid values of allocatable:
       mold=name
    """
    fmtargs = node._fmtargs

    prototype = '--NONE--'
    order = 'NPY_CORDER'
    descr = 'NULL'
    subok = '0'
    descr_code = ''

    p = re.compile('mold\s*=\s*(\w+)')
    m = p.match(allocatable)
    if m is not None:
        moldvar = m.group(1)
        moldarg = node.ast.find_arg_by_name(moldvar)
        if moldarg is None:
            raise RuntimeError(
                "Mold argument '{}' does not exist: {}"
                .format(moldvar, allocatable))
        if 'dimension' not in moldarg.attrs:
            raise RuntimeError(
                "Mold argument '{}' must have dimension attribute: {}"
                .format(moldvar, allocatable))
        fmt = fmtargs[moldvar]['fmtpy']
        # copy from the numpy array for the argument
        prototype = fmt.py_var

        # Create Descr if types are different
        if arg.typename != moldarg.typename:
            descr = 'SHDPy_' + arg.name
            descr_code = ('PyArray_Descr * {} = '
                          'PyArray_DescrFromType({});\n'
                          .format(descr, c_to_numpy[arg.typename]))

    return (prototype, order, descr, subok), descr_code


def do_cast(lang, kind, typ, var):
    """Do cast based on language.
    kind = reinterpret, static
    reinterpret_cast or static_cast
    """
    if lang == 'c':
        return '(%s) %s' % (typ, var)
    else:
        return '%s_cast<%s>\t(%s)' % (kind, typ, var)
