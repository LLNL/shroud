#!/bin/env python3
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
Generate Python module for C++ code.

Entire library in a single header.
One Extension module per class

"""
from __future__ import print_function
from __future__ import absolute_import

from . import typemap
from . import util
from .util import wformat, append_format


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

    def wrap_library(self):
        newlibrary = self.newlibrary
        options = newlibrary.options
        fmt_library = newlibrary.fmtdict

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
            typedef = typemap.Typedef.lookup(node.name)
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
        if newlibrary.functions:
            self._push_splicer('function')
#            self._begin_class()
            self.wrap_functions(None, newlibrary.functions)
            self._pop_splicer('function')

        self.write_header(newlibrary)
        self.write_module(newlibrary)
        self.write_helper()

    def wrap_class(self, node):
        self.log.write("class {1.name}\n".format(self, node))
        name = node.name
        unname = util.un_camel(name)
        typedef = typemap.Typedef.lookup(name)

        options = node.options
        fmt_class = node.fmtdict

        node.eval_template('PY_type_filename')

        self.create_class_helper_functions(node)

        self.py_type_object_creation.append(wformat("""
// {cxx_class}
    {PY_PyTypeObject}.tp_new   = PyType_GenericNew;
    {PY_PyTypeObject}.tp_alloc = PyType_GenericAlloc;
    if (PyType_Ready(&{PY_PyTypeObject}) < 0)
        return RETVAL;
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
        append_format(self.py_type_structs, '{cxx_class} * {PY_obj};', fmt_class)
        self._create_splicer('C_object', self.py_type_structs)
        self.py_type_structs.append(-1)
        self.py_type_structs.append(wformat('}} {PY_PyObject};', fmt_class))

        # wrap methods
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
            'PyObject *{PY_to_object_func}({cxx_class} *addr)', fmt)
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

    def intent_out(self, typedef, fmt, output):
        """Create PyObject from C++ value.

        typedef - typedef of C++ variable.
        fmt - format dictionary
        post_parse - output lines.
        """

        fmt.PyObject = typedef.PY_PyObject or 'PyObject'
        fmt.PyTypeObject = typedef.PY_PyTypeObject

        cmd_list = typedef.py_statements.get('intent_out', {}).get('ctor', [])
        if cmd_list:
            # must create py_var from cxx_var.
            # XXX fmt.cxx_var = 'SH_' + fmt.c_var
            for cmd in cmd_list:
                append_format(output, cmd, fmt)
            format = 'O'
            vargs = fmt.py_var

        elif typedef.PY_ctor:
            fmt.c_var = wformat(typedef.cxx_to_c, fmt)  # if C++
            append_format(output,
                          '{PyObject} * {py_var} = ' + typedef.PY_ctor
                          + ';', fmt)
            format = 'O'
            vargs = fmt.py_var

        else:
            format = typedef.PY_format
            if format == 'O':
                raise RuntimeError(
                    "PY_format should not be 'O' for " + typedef.name)
            vargs = ''
            if typedef.PY_to_object:
                format += '&'
                vargs = typedef.PY_to_object
            vargs += wformat(typedef.cxx_to_c, fmt)  # if C++

        return format, vargs

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

        self.multi_dispatch(cls, functions)

    def wrap_function(self, cls, node):
        """Write a Python wrapper for a C++ function.

        cls  - class node or None for functions
        node - function/method node

        fmt.c_var   - name of variable in PyArg_ParseTupleAndKeywords
        fmt.cxx_var - name of variable in c++ call.
        fmt.py_var  - name of PyObject variable
        fmt.PY_used_param_args - True/False if parameter args is used
        fmt.PY_used_param_kwds - True/False if parameter kwds is used
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
        if cls:
            fmt.PY_used_param_self = True

        CXX_subprogram = node._subprogram

        ast = node.ast
        result_type = ast.typename
        is_ctor = ast.fattrs.get('_constructor', False)
        is_dtor = ast.fattrs.get('_destructor', False)
#        is_const = ast.const

        if is_ctor:   # or is_dtor:
            # XXX - have explicit delete
            # need code in __init__ and __del__
            return
        if is_dtor or node.return_this:
            result_type = 'void'
            CXX_subprogram = 'subroutine'

        result_typedef = typemap.Typedef.lookup(result_type)

        # XXX if a class, then knock off const since the PyObject
        # is not const, otherwise, use const from result.
# This has been replaced by gen_arg methods, but not sure about const.
#        if result_typedef.base == 'wrapped':
#            is_const = False
#        else:
#            is_const = None
        fmt.PY_rv_decl = ast.gen_arg_as_cxx(name=fmt.PY_result)  # return value

        PY_decl = []     # variables for function
        PY_code = []

        # arguments to PyArg_ParseTupleAndKeywords
        parse_format = []
        parse_vargs = []
        post_parse = []

        # arguments to Py_BuildValue
        build_format = []
        build_vargs = []
        post_call = []

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

        args = ast.params
        if not args:
            fmt.PY_ml_flags = 'METH_NOARGS'
        else:
            fmt.PY_ml_flags = 'METH_VARARGS|METH_KEYWORDS'
            fmt.PY_used_param_args = True
            fmt.PY_used_param_kwds = True
            arg_names = []
            arg_offsets = []
            offset = 0
            for arg in args:
                arg_name = arg.name
                fmt_arg0 = fmtargs.setdefault(arg_name, {})
                fmt_arg = fmt_arg0.setdefault('fmtpy', util.Scope(fmt))
                fmt_arg.c_var = arg_name
                fmt_arg.cxx_var = arg_name
                fmt_arg.py_var = 'SH_Py_' + arg_name
                if arg.const:
                    fmt_arg.c_const = 'const '
                else:
                    fmt_arg.c_const = ''
                if arg.is_pointer():
                    fmt_arg.c_ptr = ' *'
                    fmt_arg.cxx_deref = '->'
                else:
                    fmt_arg.c_ptr = ''
                    fmt_arg.cxx_deref = '.'
                attrs = arg.attrs

                arg_typedef = typemap.Typedef.lookup(arg.typename)
                fmt_arg.cxx_type = arg_typedef.cxx_type
                fmt_arg.cxx_decl = arg.gen_arg_as_cxx()

                py_statements = arg_typedef.py_statements
                cxx_local_var = ''
                if attrs['intent'] in ['inout', 'in']:
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
                            (len(cxx_call_list), len(post_parse),
                             ',\t '.join(cxx_call_list)))

                    parse_format.append(arg_typedef.PY_format)
                    if arg_typedef.PY_PyTypeObject:
                        # Expect object of given type
                        parse_format.append('!')
                        parse_vargs.append('&' + arg_typedef.PY_PyTypeObject)
                        arg_name = fmt_arg.py_var
                    elif arg_typedef.PY_from_object:
                        # Use function to convert object
                        parse_format.append('&')
                        parse_vargs.append(arg_typedef.PY_from_object)

                    # add argument to call to PyArg_ParseTypleAndKeywords
                    parse_vargs.append('&' + arg_name)

                    stmts = 'intent_in'  # XXX -  + attrs['intent'] ?
                    intent_blk = py_statements.get(stmts, {})

                    cxx_local_var = intent_blk.get('cxx_local_var', '')
                    if cxx_local_var:
                        fmt_arg.cxx_var = 'SH_' + fmt_arg.c_var
                    cmd_list = py_statements.get(
                        'intent_in', {}).get('post_parse', [])
                    if cmd_list:
                        for cmd in cmd_list:
                            append_format(post_parse, cmd, fmt_arg)

                if attrs['intent'] in ['inout', 'out']:
                    # output variable must be a pointer
                    # XXX - fix up for strings
                    format, vargs = self.intent_out(
                        arg_typedef, fmt_arg, post_call)
                    build_format.append(format)
                    build_vargs.append('*' + vargs)

                # argument for C++ function
                if arg_typedef.base == 'string':
                    # C++ will coerce char * to std::string
                    PY_decl.append(arg.gen_arg_as_c() + ';')
                elif arg_typedef.base == 'wrapped':
                    # defined as part of py_statements.intent_in.post_parse
                    # XXX - Not sure about intent_out
                    pass
                else:
                    PY_decl.append(arg.gen_arg_as_cxx() + ';')

                if arg_typedef.PY_PyTypeObject:
                    # A Python Object which must be converted to C++ type.
                    objtype = arg_typedef.PY_PyObject or 'PyObject'
                    PY_decl.append(objtype + ' * ' + fmt_arg.py_var + ';')
                    cxx_call_list.append(fmt_arg.cxx_var)
                elif arg_typedef.PY_from_object:
                    # already a C++ type
                    cxx_call_list.append(fmt_arg.cxx_var)
                elif cxx_local_var == 'object':
                    if arg.is_pointer():
                        cxx_call_list.append('&' + fmt_arg.cxx_var)
                    else:
                        cxx_call_list.append(fmt_arg.cxx_var)
                elif cxx_local_var == 'pointer':
                    if arg.is_pointer():
                        cxx_call_list.append(fmt_arg.cxx_var)
                    else:
                        cxx_call_list.append('*' + fmt_arg.cxx_var)
                else:
                    # convert to C++ type
                    append_format(cxx_call_list, arg_typedef.c_to_cxx, fmt_arg)

            if self.language == 'c++':
                # jump through some hoops for char ** const correctness for C++
                # warning: deprecated conversion from string constant
                #    to 'char*' [-Wwrite-strings]
                if len(arg_names) == 1:
                    PY_decl.append(
                        'const char *SH_kwcpp = "%s";' % '\\0'.join(arg_names))
                else:
                    self.break_into_continuations(
                        PY_decl, options, 'c', 1,
                        'const char *SH_kwcpp =\n"' +
                        '\\0"\n"'.join(arg_names) + '";' )
                self.break_into_continuations(
                    PY_decl, options, 'c', 1,
                    'char *SH_kw_list[] = {\n' + ',\n'.join(arg_offsets)
                    + ',\nNULL };')
            else:
                self.break_into_continuations(
                    PY_decl, options, 'c', 1,
                    'char *SH_kw_list[] = {\n"' +
                    '",\n"'.join(arg_names)
                    + '",\nNULL };')
            parse_format.extend([':', fmt.function_name])
            fmt.PyArg_format = ''.join(parse_format)
            fmt.PyArg_vargs = ',\t '.join(parse_vargs)
            self.break_into_continuations(
                PY_code, options, 'c', 1,
                wformat(
                    'if (!PyArg_ParseTupleAndKeywords'
                    '({PY_param_args}, {PY_param_kwds}, '
                    '"{PyArg_format}", SH_kw_list,'
                    '\n{PyArg_vargs}))', fmt))
            PY_code.extend(['{', 1, 'return NULL;', -1, '}'])

        if cls:
            #  template = '{c_const}{cxx_class} *{C_this}obj = static_cast<{c_const}{cxx_class} *>(static_cast<{c_const}void *>({C_this}));'
            #  fmt_func.C_object = wformat(template, fmt_func)
            # call method syntax
            fmt.PY_this_call = wformat('self->{PY_obj}->', fmt)
        else:
            fmt.PY_this_call = ''  # call function syntax

        # call with all arguments
        default_calls.append(
            (len(cxx_call_list),  len(post_parse), ',\t '.join(cxx_call_list)))

        # If multiple calls, declare return value once
        # Else delare on call line.
        if found_default:
            fmt.PY_rv_asgn = 'rv = '
            PY_code.append('switch (SH_nargs) {')
        else:
            fmt.PY_rv_asgn = fmt.PY_rv_decl + ' = '
        need_rv = False

        for nargs, len_post_parse, call_list in default_calls:
            if found_default:
                PY_code.append('case %d:' % nargs)
                PY_code.append(1)

            fmt.PY_call_list = call_list
            PY_code.extend(post_parse[:len_post_parse])

            if is_dtor:
                append_format(PY_code, 'delete self->{PY_obj};', fmt)
                append_format(PY_code, 'self->{PY_obj} = NULL;', fmt)
            elif CXX_subprogram == 'subroutine':
                line = wformat(
                    '{PY_this_call}{function_name}({PY_call_list});', fmt)
                self.break_into_continuations(
                    PY_code, options, 'c', 1, line)
            else:
                need_rv = True
                line = wformat(
                    '{PY_rv_asgn}{PY_this_call}{function_name}({PY_call_list});',
                    fmt)
                self.break_into_continuations(
                    PY_code, options, 'c', 1, line)

            if node.PY_error_pattern:
                lfmt = util.Scope(fmt)
                lfmt.c_var = fmt.PY_result
                lfmt.cxx_var = fmt.PY_result
                append_format(PY_code,
                              self.patterns[node.PY_error_pattern], lfmt)

            if found_default:
                PY_code.append('break;')
                PY_code.append(-1)
        if found_default:
            # PY_code.append('default:')
            # PY_code.append(1)
            # PY_code.append('continue;')  # XXX raise internal error
            # PY_code.append(-1)
            PY_code.append('}')
        else:
            need_rv = False

        if need_rv:
            PY_decl.append(fmt.PY_rv_decl + ';')
        if len(PY_decl):
            PY_decl.append('')

        # Compute return value
        if CXX_subprogram == 'function':
            if ast.is_pointer():
                fmt.cxx_deref = '->'
            else:
                fmt.cxx_deref = '.'
            fmt.c_var = fmt.PY_result
            fmt.cxx_var = fmt.PY_result
            fmt.py_var = 'SH_Py_' + fmt.cxx_var
            format, vargs = self.intent_out(result_typedef, fmt, PY_code)
            # Add result to front of result tuple
            build_format.insert(0, format)
            build_vargs.insert(0, vargs)

        PY_code.extend(post_call)

        # may be multiple return values using intent(OUT)
        fmt.PyArg_format = ''.join(build_format)
        fmt.PyArg_vargs = ', '.join(build_vargs)
        if not build_format:
            PY_code.append('Py_RETURN_NONE;')
        elif len(build_format) > 1:
            # return tuple
            PY_code.append(wformat(
                'return Py_BuildValue("({PyArg_format})", {PyArg_vargs});',
                fmt))
        elif build_format[0] == 'O':
            # return a single object already created
            fmt.py_var = build_vargs[0]
            append_format(PY_code, 'return (PyObject *) {py_var};', fmt)
        else:
            # create object
            PY_code.append(wformat(
                'return Py_BuildValue("{PyArg_format}", {PyArg_vargs});', fmt))

        PY_impl = [1] + PY_decl + PY_code + [-1]

        node.eval_template('PY_name_impl')

        expose = True
        if len(self.overloaded_methods[ast.name]) > 1:
            # Only expose a multi-dispatch name, not each overload
            expose = False
        elif found_default:
            # Only one wrapper to deal with default arugments.
            # [C creates a wrapper per default argument]
            fmt = util.Scope(fmt)
            fmt.function_suffix = ''

        self.create_method(cls, expose, fmt, PY_impl)

    def create_method(self, cls, expose, fmt, PY_impl):
        """Format the function.
        cls     = True if class
        expose  = True if expose to user
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

        body.extend([
                '',
                'static PyObject *',
                wformat('{PY_name_impl}(', fmt)
                ])
        if fmt.PY_used_param_self:
            body.append(wformat(
                '  {PY_PyObject} *{PY_param_self},', fmt))
        else:
            body.append(wformat(
                '  PyObject *,  // {PY_param_self} unused', fmt))
        if fmt.PY_used_param_args:
            body.append(wformat(
                '  PyObject *{PY_param_args},', fmt))
        else:
            body.append(wformat(
                '  PyObject *,  // {PY_param_args} unused', fmt))
        if fmt.PY_used_param_args:
            body.append(wformat(
                '  PyObject *{PY_param_kwds})', fmt))
        else:
            body.append(wformat(
                '  PyObject *)  // {PY_param_kwds} unused', fmt))

        body.append('{')
# use function_suffix in splicer name since a single C++ function may
# produce several methods.
# XXX - make splicer name customizable?
#        self._create_splicer(fmt.function_name, self.PyMethodBody, PY_impl)
        self._create_splicer(fmt.underscore_name + fmt.function_suffix,
                             self.PyMethodBody, PY_impl)
        self.PyMethodBody.append('}')

        if expose is True:
            # default name
            self.PyMethodDef.append(
                wformat('{{"{function_name}{function_suffix}", '
                        '(PyCFunction){PY_name_impl}, '
                        '{PY_ml_flags}, '
                        '{PY_name_impl}__doc__}},', fmt))
#        elif expose is not False:
#            # override name
#            fmt = util.Scope(fmt)
#            fmt.expose = expose
#            self.PyMethodDef.append( wformat('{{"{expose}", (PyCFunction){PY_name_impl}, {PY_ml_flags}, {PY_name_impl}__doc__}},', fmt))

    def write_tp_func(self, node, fmt, fmt_type, output):
        # fmt is a dictiony here.
        # update with type function names
        # type bodies must be filled in by user, no real way to guess
        PyObj = fmt.PY_PyObject
        selected = node.python.get('type', [])

        # Dictionary of methods for bodies
        default_body = dict(
            richcompare=self.not_implemented
        )

        self._push_splicer('type')
        for typename in typenames:
            tp_name = 'tp_' + typename
            if typename not in selected:
                fmt_type[tp_name] = '0'
                continue
            func_name = wformat('{PY_prefix}{cxx_class}_tp_', fmt) + typename
            fmt_type[tp_name] = func_name
            tup = typefuncs[typename]
            output.append('static ' + tup[0])
            output.append(('{name} ' + tup[1])
                          .format(name=func_name, object=PyObj))
            output.append('{')
            default = default_body.get(typename, self.not_implemented_error)
            default = default(typename, tup[2])
            self._create_splicer(typename, output, default)
            output.append('}')
        self._pop_splicer('type')

    def write_extension_type(self, library, node):
        fmt = node.fmtdict
        fname = fmt.PY_type_filename

        output = []

        output.append(wformat('#include "{PY_header_filename}"', fmt))
        self._push_splicer('impl')
        self._create_splicer('include', output)
        self.namespace(library, node, 'begin', output)
        self._create_splicer('C_definition', output)
        self._create_splicer('additional_methods', output)
        self._pop_splicer('impl')

        fmt_type = dict(
            PY_module_name=fmt.PY_module_name,
            PY_PyObject=fmt.PY_PyObject,
            PY_PyTypeObject=fmt.PY_PyTypeObject,
            cxx_class=fmt.cxx_class,
            )
        self.write_tp_func(node, fmt, fmt_type, output)

        output.extend(self.PyMethodBody)

        self._push_splicer('impl')
        self._create_splicer('after_methods', output)
        self._pop_splicer('impl')

        fmt_type['tp_methods'] = wformat('{PY_prefix}{cxx_class}_methods', fmt)
        output.append(
            wformat('static PyMethodDef {tp_methods}[] = {{', fmt_type))
        output.extend(self.PyMethodDef)
        self._create_splicer('PyMethodDef', output)
        output.append('{NULL,   (PyCFunction)NULL, 0, NULL}'
                      '            /* sentinel */')
        output.append('};')

        output.append(wformat(PyTypeObject_template, fmt_type))
        self.namespace(library, node, 'end', output)

        self.write_output_file(fname, self.config.python_dir, output)

    def multi_dispatch(self, cls, methods):
        """Look for overloaded methods.
        When found, create a method which will call each of the
        overloaded methods looking for the one which will accept
        the given arguments.
        """
        for method, methods in self.overloaded_methods.items():
            if len(methods) < 2:
                continue  # not overloaded

            fmt_func = methods[0].fmtdict
            fmt = util.Scope(fmt_func)
            fmt.function_suffix = ''
            fmt.PY_doc_string = 'documentation'
            fmt.PY_ml_flags = 'METH_VARARGS|METH_KEYWORDS'
            fmt.PY_used_param_self = True
            fmt.PY_used_param_args = True
            fmt.PY_used_param_kwds = True

            body = []
            body.append(1)
            body.append('Py_ssize_t SH_nargs = 0;')
            body.extend([
                    'if (args != NULL) SH_nargs += PyTuple_Size(args);',
                    'if (kwds != NULL) SH_nargs += PyDict_Size(args);',
                    ])
            body.append('PyObject *rvobj;')

            for overload in methods:
                if overload._nargs:
                    body.append('if (SH_nargs >= %d && SH_nargs <= %d) {'
                                % overload._nargs)
                else:
                    body.append('if (SH_nargs == %d) {' %
                                len(overload.ast.params))
                body.append(1)
                append_format(body,
                              'rvobj = {PY_name_impl}(self, args, kwds);',
                              overload.fmtdict)
                body.append('if (!PyErr_Occurred()) {')
                body.append(1)
                body.append('return rvobj;')
                body.append(-1)
                body.append('} else if (! PyErr_ExceptionMatches'
                            '(PyExc_TypeError)) {')
                body.append(1)
                body.append('return rvobj;')
                body.append(-1)
                body.append('}')
                body.append('PyErr_Clear();')
                body.append(-1)
                body.append('}')

            body.append('PyErr_SetString(PyExc_TypeError, '
                        '"wrong arguments multi-dispatch");')
            body.append('return NULL;')
            body.append(-1)

            methods[0].eval_template('PY_name_impl', fmt=fmt)

            self.create_method(cls, True, fmt, body)

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

        output.extend([
                '#include <Python.h>',
                '#if PY_MAJOR_VERSION >= 3',
                '#define IS_PY3K',
                '#endif'])

        for include in node.cxx_header.split():
            output.append('#include "%s"' % include)

        self._push_splicer('header')
        self._create_splicer('include', output)
        self.namespace(node, None, 'begin', output)
        output.extend(self.py_type_extern)
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

#ifdef __cplusplus
extern "C" {{
#endif
#ifdef IS_PY3K
#define MOD_INITBASIS PyInit_{PY_module_name}
#else
#define MOD_INITBASIS init{PY_module_name}
#endif
PyMODINIT_FUNC MOD_INITBASIS(void);
#ifdef __cplusplus
}}
#endif
""", fmt))
        self.namespace(node, None, 'end', output)
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
        output.append('')
        self._create_splicer('include', output)
        self.namespace(node, None, 'begin', output)
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
        output.extend(self.py_type_object_creation)
        output.append(wformat(module_middle2, fmt))
        self._create_splicer('C_init_body', output)
        output.append(wformat(module_end, fmt))
        self.namespace(node, None, 'end', output)

        self.write_output_file(fname, self.config.python_dir, output)

    def write_helper(self):
        node = self.newlibrary
        fmt = node.fmtdict
        output = []
        output.append(wformat('#include "{PY_header_filename}"', fmt))
        self.namespace(node, None, 'begin', output)
        output.extend(self.py_helper_definition)
        output.append('')
        output.extend(self.py_helper_functions)
        self.namespace(node, None, 'end', output)
        self.write_output_file(
            fmt.PY_helper_filename, self.config.python_dir, output)

    def not_implemented_error(self, msg, ret):
        '''A standard splicer for unimplemented code
        ret is the return value (NULL or -1)
        '''
        return [
            "    PyErr_SetString(PyExc_NotImplementedError, \"%s\");" % msg,
            "    return %s;" % ret
            ]

    def not_implemented(self, msg, ret):
        '''A standard splicer for rich comparison
        '''
        return [
            'Py_INCREF(Py_NotImplemented);',
            'return Py_NotImplemented;'
            ]


# --- Python boiler plate

typenames = [
    'dealloc', 'print', 'compare',
    'getattr', 'setattr',  # deprecated
    'getattro', 'setattro',
    'repr', 'hash', 'call', 'str',
    'init', 'alloc', 'new', 'free', 'del',
    'richcompare'
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

PyTypeObject_template = """
static char {cxx_class}__doc__[] =
"virtual class"
;

/* static */
PyTypeObject {PY_PyTypeObject} = {{
        PyVarObject_HEAD_INIT(NULL, 0)
        "{PY_module_name}.{cxx_class}",                       /* tp_name */
        sizeof({PY_PyObject}),         /* tp_basicsize */
        0,                              /* tp_itemsize */
        /* Methods to implement standard operations */
        (destructor){tp_dealloc},                 /* tp_dealloc */
        (printfunc){tp_print},                   /* tp_print */
        (getattrfunc){tp_getattr},                 /* tp_getattr */
        (setattrfunc){tp_setattr},                 /* tp_setattr */
#ifdef IS_PY3K
        0,                               /* tp_reserved */
#else
        (cmpfunc){tp_compare},                     /* tp_compare */
#endif
        (reprfunc){tp_repr},                    /* tp_repr */
        /* Method suites for standard classes */
        0,                              /* tp_as_number */
        0,                              /* tp_as_sequence */
        0,                              /* tp_as_mapping */
        /* More standard operations (here for binary compatibility) */
        (hashfunc){tp_hash},                    /* tp_hash */
        (ternaryfunc){tp_call},                 /* tp_call */
        (reprfunc){tp_str},                    /* tp_str */
        (getattrofunc){tp_getattro},                /* tp_getattro */
        (setattrofunc){tp_setattro},                /* tp_setattro */
        /* Functions to access object as input/output buffer */
        0,                              /* tp_as_buffer */
        /* Flags to define presence of optional/expanded features */
        Py_TPFLAGS_DEFAULT,             /* tp_flags */
        {cxx_class}__doc__,         /* tp_doc */
        /* Assigned meaning in release 2.0 */
        /* call function for all accessible objects */
        (traverseproc)0,                /* tp_traverse */
        /* delete references to contained objects */
        (inquiry)0,                     /* tp_clear */
        /* Assigned meaning in release 2.1 */
        /* rich comparisons */
        (richcmpfunc){tp_richcompare},                 /* tp_richcompare */
        /* weak reference enabler */
        0,                              /* tp_weaklistoffset */
        /* Added in release 2.2 */
        /* Iterators */
        (getiterfunc)0,                 /* tp_iter */
        (iternextfunc)0,                /* tp_iternext */
        /* Attribute descriptor and subclassing stuff */
        {tp_methods},                             /* tp_methods */
        0,                              /* tp_members */
        0,                             /* tp_getset */
        0,                              /* tp_base */
        0,                              /* tp_dict */
        (descrgetfunc)0,                /* tp_descr_get */
        (descrsetfunc)0,                /* tp_descr_set */
        0,                              /* tp_dictoffset */
        (initproc){tp_init},                   /* tp_init */
        (allocfunc){tp_alloc},                  /* tp_alloc */
        (newfunc){tp_new},                    /* tp_new */
        (freefunc){tp_free},                   /* tp_free */
        (inquiry)0,                     /* tp_is_gc */
        0,                              /* tp_bases */
        0,                              /* tp_mro */
        0,                              /* tp_cache */
        0,                              /* tp_subclasses */
        0,                              /* tp_weaklist */
        (destructor){tp_del},                 /* tp_del */
        0,                              /* tp_version_tag */
#ifdef IS_PY3K
        (destructor)0,                  /* tp_finalize */
#endif
}};"""


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

#ifdef IS_PY3K
#define GETSTATE(m) ((struct module_state*)PyModule_GetState(m))
#else
#define GETSTATE(m) (&_state)
static struct module_state _state;
#endif

#ifdef IS_PY3K
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

#ifdef __cplusplus
extern "C" {{
#endif
PyMODINIT_FUNC
MOD_INITBASIS(void)
{{
    PyObject *m = NULL;
    const char * error_name = "{library_lower}.Error";
"""

module_middle = """

    /* Create the module and add the functions */
#ifdef IS_PY3K
    m = PyModule_Create(&moduledef);
#else
    m = Py_InitModule4("{PY_module_name}", {PY_prefix}methods,
                       {PY_prefix}_doc__,
                       (PyObject*)NULL,PYTHON_API_VERSION);
#endif
    if (m == NULL)
        return RETVAL;
    struct module_state *st = GETSTATE(m);
"""

module_middle2 = """
    {PY_prefix}error_obj = PyErr_NewException((char *) error_name, NULL, NULL);
    if ({PY_prefix}error_obj == NULL)
        return RETVAL;
    st->error = {PY_prefix}error_obj;
    PyModule_AddObject(m, "Error", st->error);
"""

module_end = """
    /* Check for errors */
    if (PyErr_Occurred())
        Py_FatalError("can't initialize module {PY_module_name}");
    return RETVAL;
}}
#ifdef __cplusplus
}}
#endif
"""
