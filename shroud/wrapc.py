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
Generate C bindings for C++ classes

"""
from __future__ import print_function
from __future__ import absolute_import

import os

from . import declast
from . import typemap
from . import whelpers
from . import util
from .util import append_format

wformat = util.wformat


class Wrapc(util.WrapperMixin):
    """Generate C bindings for C++ classes

    """
    def __init__(self, newlibrary, config, splicers):
        self.newlibrary = newlibrary
        self.patterns = newlibrary.patterns
        self.language = newlibrary.language
        self.config = config
        self.log = config.log
        self._init_splicer(splicers)
        self.comment = '//'
        self.doxygen_begin = '/**'
        self.doxygen_cont = ' *'
        self.doxygen_end = ' */'

    def _begin_output_file(self):
        """Start a new class for output"""
        # forward declarations of C++ class as opaque C struct.
        self.header_forward = {}
        # include files required by typedefs
        self.header_typedef_include = {}
        # headers needed by implementation, i.e. helper functions
        self.header_impl_include = {}
        self.header_proto_c = []
        self.impl = []
        self.c_helper = {}

    def wrap_library(self):
        newlibrary = self.newlibrary
        fmt_library = newlibrary.fmtdict

        self._push_splicer('class')
        for node in newlibrary.classes:
            if not node.options.wrap_c:
                continue
            self._push_splicer(node.name)
            self.write_file(newlibrary, node)
            self._pop_splicer(node.name)
        self._pop_splicer('class')

        if self.newlibrary.functions:
            self.write_file(newlibrary, None)

    def write_file(self, library, cls):
        """Write a file for the library and its functions or
        a class and its methods.
        """
        node = cls or library
        fmt = node.fmtdict
        self._begin_output_file()
        if cls:
            self.wrap_class(cls)
        else:
            self.wrap_functions(library)
        c_header = fmt.C_header_filename
        c_impl = fmt.C_impl_filename
        self.write_header(library, cls, c_header)
        self.write_impl(library, cls, c_header, c_impl)

    def wrap_functions(self, library):
        # worker function for write_file
        self._push_splicer('function')
        for node in library.functions:
            self.wrap_function(None, node)
        self._pop_splicer('function')

    def write_header(self, library, cls, fname):
        """ Write header file for a library node or a class node.
        """
        guard = fname.replace(".", "_").upper()
        node = cls or library
        options = node.options

        # If no C wrappers are required, do not write the file
        write_file = False
        output = []

        if options.doxygen:
            self.write_doxygen_file(output, fname, library, cls)

        output.extend([
                '// For C users and %s implementation' % self.language.upper(),
                '',
                '#ifndef %s' % guard,
                '#define %s' % guard,
                ])
        if cls and cls.cpp_if:
            output.append('#' + node.cpp_if)

        # headers required by typedefs
        if self.header_typedef_include:
            # output.append('// header_typedef_include')
            output.append('')
            headers = self.header_typedef_include.keys()
            self.write_headers(headers, output)

        if self.language == 'c++':
            output.append('')
            if self._create_splicer('CXX_declarations', output):
                write_file = True
            output.extend([
                    '',
                    '#ifdef __cplusplus',
                    'extern "C" {',
                    '#endif'
                    ])
        output.extend([
                '',
                '// declaration of wrapped types'
                ])
        names = sorted(self.header_forward.keys())
        for name in names:
            write_file = True
            output.append(
                'struct s_{C_type_name};\n'
                'typedef struct s_{C_type_name} {C_type_name};'.
                format(C_type_name=name))
        output.append('')
        if self._create_splicer('C_declarations', output):
            write_file = True
        if self.header_proto_c:
            write_file = True
            output.extend(self.header_proto_c)
        if self.language == 'c++':
            output.extend([
                    '',
                    '#ifdef __cplusplus',
                    '}',
                    '#endif'
                    ])
        if cls and cls.cpp_if:
            output.append('#endif  // ' + node.cpp_if)
        output.extend([
                '',
                '#endif  // ' + guard
                ])

        if write_file:
            self.config.cfiles.append(
                os.path.join(self.config.c_fortran_dir, fname))
            self.write_output_file(fname, self.config.c_fortran_dir, output)

    def write_impl(self, library, cls, hname, fname):
        """Write implementation
        """
        node = cls or library
        options = node.options

        # If no C wrappers are required, do not write the file
        write_file = False
        output = []
        if cls and cls.cpp_if:
            output.append('#' + node.cpp_if)

        # Insert any helper functions needed
        helper_source = []
        if self.c_helper:
            helperdict = whelpers.find_all_helpers('c', self.c_helper)
            helpers = sorted(self.c_helper)
            if self.language == 'c':
                lang_header = 'c_header'
                lang_source = 'c_source'
            else:
                lang_header = 'cxx_header'
                lang_source = 'cxx_source'
            for helper in helpers:
                helper_info = helperdict[helper]
                if lang_header in helper_info:
                    for include in helper_info[lang_header].split():
                        self.header_impl_include[include] = True
                if lang_source in helper_info:
                    helper_source.append(helper_info[lang_source])
                elif 'source' in helper_info:
                    helper_source.append(helper_info['source'])

        output.append('#include "%s"' % hname)

        # Use headers from class if they exist or else library
        if cls and cls.cxx_header:
            for include in cls.cxx_header.split():
                self.header_impl_include[include] = True
        else:
            for include in library.cxx_header.split():
                self.header_impl_include[include] = True

        # headers required by implementation
        if self.header_impl_include:
            headers = self.header_impl_include.keys()
            self.write_headers(headers, output)

        if helper_source:
            write_file = True
            output.extend(helper_source)

        self.namespace(library, cls, 'begin', output)
        if self.language == 'c++':
            output.append('')
            if self._create_splicer('CXX_definitions', output):
                write_file = True
            output.append('\nextern "C" {')
        output.append('')
        if self._create_splicer('C_definitions', output):
            write_file = True
        if self.impl:
            write_file = True
            output.extend(self.impl)

        if self.language == 'c++':
            output.append('')
            output.append('}  // extern "C"')
        self.namespace(library, cls, 'end', output)

        if cls and cls.cpp_if:
            output.append('#endif  // ' + node.cpp_if)

        if write_file:
            self.config.cfiles.append(
                os.path.join(self.config.c_fortran_dir, fname))
            self.write_output_file(fname, self.config.c_fortran_dir, output)

    def write_headers(self, headers, output):
        for header in sorted(headers):
            if header[0] == '<':
                output.append('#include %s' % header)
            else:
                output.append('#include "%s"' % header)

    def wrap_class(self, node):
        self.log.write("class {1.name}\n".format(self, node))
        name = node.name
        typedef = typemap.Typedef.lookup(name)
        cname = typedef.c_type

        fmt_class = node.fmtdict
        # call method syntax
        fmt_class.CXX_this_call = fmt_class.CXX_this + '->'

        # create a forward declaration for this type
        self.header_forward[cname] = True

        self._push_splicer('method')
        for method in node.functions:
            self.wrap_function(node, method)
        self._pop_splicer('method')

    def wrap_function(self, cls, node):
        """
        Wrap a C++ function with C
        cls  - class node or None for functions
        node - function/method node
        """
        options = node.options
        if not options.wrap_c:
            return

        if cls:
            cls_function = 'method'
        else:
            cls_function = 'function'
        self.log.write("C {0} {1.declgen}\n".format(cls_function, node))

        fmt_func = node.fmtdict
        fmtargs = node._fmtargs

        if self.language == 'c' or options.get('C_extern_C',False):
            # Fortran can call C directly and only needs wrappers when code is
            # inserted. For example, precall or postcall.
            need_wrapper = False
        else:
            # C++ will need C wrappers to deal with name mangling.
            need_wrapper = True
        if self.language == 'c':
            lang_header = 'c_header'
        else:
            lang_header = 'cxx_header'

        # Look for C++ routine to wrap
        # Usually the same node unless it is generated (i.e. bufferified)
        CXX_node = node
        generated = []
        if CXX_node._generated:
            generated.append(CXX_node._generated)
        while CXX_node._PTR_C_CXX_index is not None:
            CXX_node = self.newlibrary.function_index[CXX_node._PTR_C_CXX_index]
            if CXX_node._generated:
                generated.append(CXX_node._generated)
        CXX_result = CXX_node.ast
        CXX_subprogram = CXX_node._subprogram

        # C return type
        ast = node.ast
        result_type = ast.typename
        subprogram = node._subprogram
        generated_suffix = ''
        if node._generated == 'arg_to_buffer':
            generated_suffix = '_buf'

        result_typedef = typemap.Typedef.lookup(result_type)
        result_is_const = ast.const
        is_ctor = CXX_result.fattrs.get('_constructor', False)
        is_dtor = CXX_result.fattrs.get('_destructor', False)
        is_const = ast.func_const

        # C++ functions which return 'this',
        # are easier to call from Fortran if they are subroutines.
        # There is no way to chain in Fortran:  obj->doA()->doB();
        if node.return_this or is_dtor:
            CXX_subprogram = 'subroutine'

        if result_typedef.c_header:
            # include any dependent header in generated header
            self.header_typedef_include[result_typedef.c_header] = True
        if result_typedef.cxx_header:
            # include any dependent header in generated source
            self.header_impl_include[result_typedef.cxx_header] = True
        if result_typedef.forward:
            # create forward references for other types being wrapped
            # i.e. This method returns a wrapped type
            self.header_forward[result_typedef.c_type] = True

        if result_is_const:
            fmt_func.c_const = 'const '
        else:
            fmt_func.c_const = ''

        return_lang = '{cxx_var}'  # Assume C and C++ types are compatiable
        if CXX_subprogram == 'subroutine':
            fmt_result = fmt_func
            fmt_pattern = fmt_func
        else:
            fmt_result0 = node._fmtresult
            fmt_result = fmt_result0.setdefault('fmtc', util.Scope(fmt_func))
            fmt_result.cxx_var = fmt_func.C_result
            fmt_result.cxx_rv_decl = CXX_result.gen_arg_as_cxx(name=fmt_func.C_result)
            if CXX_result.is_pointer():
                fmt_result.cxx_deref = '->'
            else:
                fmt_result.cxx_deref = '.'
            fmt_pattern = fmt_result

        proto_list = []
        call_list = []
        if cls:
            need_wrapper = True
            # object pointer
            rvast = declast.create_this_arg(fmt_func.C_this, cls.name, is_const)
            if not is_ctor:
                arg = rvast.gen_arg_as_c()
                proto_list.append(arg)

        # indicate which argument contains function result, usually none
        result_arg = None
        pre_call = []      # list of temporary variable declarations
        post_call = []

        if cls and not is_ctor:
            if is_const:
                fmt_func.c_const = 'const '
            else:
                fmt_func.c_const = ''
            fmt_func.c_ptr = ' *'
            fmt_func.c_var = fmt_func.C_this
            # LHS is class' cxx_to_c
            cls_typedef = typemap.Typedef.lookup(cls.name)
            append_format(pre_call, 
                          '{c_const}{cxx_class} *{CXX_this} = ' +
                          cls_typedef.c_to_cxx + ';', fmt_func)

#    c_var      - argument to C function  (wrapper function)
#    c_var_trim - variable with trimmed length of c_var
#    c_var_len  - variable with length of c_var
#    cxx_var    - argument to C++ function  (wrapped function).
#                 Usually same as c_var but may be a new local variable
#                 or the funtion result variable.

        for arg in ast.params:
            arg_name = arg.name
            fmt_arg0 = fmtargs.setdefault(arg_name, {})
            fmt_arg = fmt_arg0.setdefault('fmtc', util.Scope(fmt_func))
            c_attrs = arg.attrs
            arg_typedef, c_statements = typemap.lookup_c_statements(arg)
            if 'template' in c_attrs:
                fmt_arg.cxx_T = c_attrs['template']

            fmt_arg.c_var = arg_name

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
            fmt_arg.cxx_type = arg_typedef.cxx_type

            proto_list.append(arg.gen_arg_as_c())

            if c_attrs.get('_is_result', False):
                arg_call = False
                fmt_arg.cxx_var = fmt_arg.C_result
                fmt_pattern = fmt_arg
                result_arg = arg
                stmts = 'result' + generated_suffix
                need_wrapper = True
                if CXX_result.is_pointer():
                    fmt_arg.cxx_deref = '->'
                else:
                    fmt_arg.cxx_deref = '.'
            else:
                arg_call = arg
                fmt_arg.cxx_var = fmt_arg.c_var      # name in c++ call.
                stmts = 'intent_' + c_attrs['intent'] + generated_suffix

            intent_blk = c_statements.get(stmts, {})

            # Add implied buffer arguments to prototype
            for buf_arg in intent_blk.get('buf_args', []):
                need_wrapper = True
                if buf_arg == 'size':
                    fmt_arg.c_var_size = c_attrs['size']
                    append_format(proto_list, 'long {c_var_size}', fmt_arg)
                elif buf_arg == 'len_trim':
                    fmt_arg.c_var_trim = c_attrs['len_trim']
                    append_format(proto_list, 'int {c_var_trim}', fmt_arg)
                elif buf_arg == 'len':
                    fmt_arg.c_var_len = c_attrs['len']
                    append_format(proto_list, 'int {c_var_len}', fmt_arg)

            # Add any code needed for intent(IN).
            # Usually to convert types.
            # For example, convert char * to std::string
            # Skip input arguments generated by F_string_result_as_arg
            cxx_local_var = intent_blk.get('cxx_local_var', '')
            if cxx_local_var:
                fmt_arg.cxx_var = 'SH_' + fmt_arg.c_var
                if cxx_local_var == 'object':
                    fmt_arg.cxx_deref = '.'
                elif cxx_local_var == 'pointer':
                    fmt_arg.cxx_deref = '->'

            # Add code for intent of argument
            # pre_call.append('// intent=%s' % intent)
            cmd_list = intent_blk.get('pre_call', [])
            if cmd_list:
                need_wrapper = True
                for cmd in cmd_list:
                    append_format(pre_call, cmd, fmt_arg)

            cmd_list = intent_blk.get('post_call', [])
            if cmd_list:
                need_wrapper = True
                for cmd in cmd_list:
                    append_format(post_call, cmd, fmt_arg)

            if 'c_helper' in intent_blk:
                for helper in intent_blk['c_helper'].split():
                    self.c_helper[helper] = True

            cxx_header = intent_blk.get(lang_header, None)
            # include any dependent header in generated source
            if cxx_header:
                for h in cxx_header.split():
                    self.header_impl_include[h] = True

            if arg_call:
                # Skips result_as_arg argument
                if cxx_local_var == 'object':
                    if arg.is_pointer():
                        call_list.append('&' + fmt_arg.cxx_var)
                    else:
                        call_list.append(fmt_arg.cxx_var)
                elif cxx_local_var == 'pointer':
                    if arg.is_pointer():
                        call_list.append(fmt_arg.cxx_var)
                    else:
                        call_list.append('*' + fmt_arg.cxx_var)
                else:
                    # convert C argument to C++
                    append_format(call_list, arg_typedef.c_to_cxx, fmt_arg)

            if arg_typedef.c_header:
                # include any dependent header in generated header
                self.header_typedef_include[arg_typedef.c_header] = True
            if arg_typedef.cxx_header:
                # include any dependent header in generated source
                self.header_impl_include[arg_typedef.cxx_header] = True
            if arg_typedef.forward:
                # create forward references for other types being wrapped
                # i.e. This argument is another wrapped type
                self.header_forward[arg_typedef.c_type] = True
        fmt_func.C_call_list = ', '.join(call_list)

        fmt_func.C_prototype = options.get('C_prototype', ', '.join(proto_list))

        if node.return_this:
            fmt_func.C_return_type = 'void'
        elif is_dtor:
            fmt_func.C_return_type = 'void'
        elif fmt_func.C_custom_return_type:
            pass
        else:
            fmt_func.C_return_type = ast.gen_arg_as_c(name=None)

        post_call_pattern = []
        if node.C_error_pattern is not None:
            C_error_pattern = node.C_error_pattern + generated_suffix
            if C_error_pattern in self.patterns:
                post_call_pattern.append('// C_error_pattern')
                append_format(
                    post_call_pattern, self.patterns[C_error_pattern], fmt_pattern)
        if post_call_pattern:
            need_wrapper = True
            fmt_func.C_post_call_pattern = '\n'.join(post_call_pattern)

        # generate the C body
        C_return_code = 'return;'
        if is_ctor:
            fmt_func.C_call_code = wformat('{cxx_rv_decl} = new {cxx_class}'
                           '({C_call_list});', fmt_result)
            call_code = [ fmt_result.cxx_rv_decl, ' = new ',
                          fmt_result.cxx_class, call_list ]
            C_return_code = ('return {};'.format(
                wformat(result_typedef.cxx_to_c, fmt_result)))
        elif is_dtor:
            fmt_func.C_call_code = 'delete %s;' % fmt_func.CXX_this
            call_code = [ 'delete %s' % fmt_func.CXX_this ]
        elif CXX_subprogram == 'subroutine':
            fmt_func.C_call_code = wformat(
                '{CXX_this_call}{function_name}'
                '{CXX_template}({C_call_list});',
                fmt_func)
            call_code = [ fmt_func.CXX_this_call, fmt_func.function_name,
                          fmt_func.CXX_template, call_list ]
        else:
            fmt_func.C_call_code = wformat(
                '{cxx_rv_decl} = {CXX_this_call}{function_name}'
                '{CXX_template}({C_call_list});',
                fmt_result)
            call_code = [ fmt_result.cxx_rv_decl, ' = ',
                          fmt_result.CXX_this_call, fmt_result.function_name,
                          fmt_result.CXX_template, call_list ]

            if result_arg is None:
                # The result is not passed back in an argument
                c_statements = result_typedef.c_statements
                intent_blk = c_statements.get('result', {})
                if result_typedef.cxx_to_c != '{cxx_var}':
                    # Make intermediate c_var value if a conversion
                    # is required i.e. not the same as cxx_var.
                    have_c_local_var = True
                else:
                    have_c_local_var = intent_blk.get('c_local_var', False)
                    if have_c_local_var:
                        raise RuntimeError  # XXX dead code
                if have_c_local_var:
                    # XXX need better mangling than 'X'
                    fmt_result.c_var = 'X' + fmt_func.C_result
                    fmt_result.c_rv_decl = CXX_result.gen_arg_as_c(
                        name=fmt_result.c_var)
                    fmt_result.c_val = wformat(result_typedef.cxx_to_c, fmt_result)
                    append_format(post_call, '{c_rv_decl} = {c_val};', fmt_result)
                    return_lang = '{c_var}'

                cmd_list = intent_blk.get('post_call', [])
                for cmd in cmd_list:
                    append_format(post_call, cmd, fmt_result)
                # XXX release rv if necessary
                if 'c_helper' in intent_blk:
                    for helper in intent_blk['c_helper'].split():
                        self.c_helper[helper] = True

            if subprogram == 'function':
                # Note: A C function may be converted into a Fortran subroutine
                # subprogram when the result is returned in an argument.
                C_return_code = 'return {};'.format(
                    wformat(return_lang, fmt_result))

        if fmt_func.inlocal('C_finalize' + generated_suffix):
            # maybe check C_finalize up chain for accumulative code
            # i.e. per class, per library.
            finalize_line = fmt_func.get('C_finalize' + generated_suffix)
            need_wrapper = True
            post_call.append('{')
            post_call.append('    // C_finalize')
            util.append_format_indent(post_call, finalize_line, fmt_func)
            post_call.append('}')

        if fmt_func.inlocal('C_return_code'):
            need_wrapper = True
            C_return_code = wformat(fmt_func.C_return_code, fmt_func)
        else:
            fmt_func.C_return_code = C_return_code

        if pre_call:
            fmt_func.C_pre_call = '\n'.join(pre_call)
        if post_call:
            fmt_func.C_post_call = '\n'.join(post_call)

        splicer_code = self.splicer_stack[-1].get(fmt_func.function_name, None)
        if fmt_func.inlocal('C_code'):
            need_wrapper = True
            C_code = [1, wformat(fmt_func.C_code, fmt_func), -1]
        elif splicer_code:
            need_wrapper = True
            C_code = splicer_code
        else:
            # copy-out values, clean up
            C_code = [1]
            C_code.extend(pre_call)
            C_code.append(self.continued_line('', ';', 1, *call_code))

            C_code.extend(post_call_pattern)
            C_code.extend(post_call)
            C_code.append(fmt_func.C_return_code)
            C_code.append(-1)

        if need_wrapper:
            self.header_proto_c.append('')
            if node.cpp_if:
                self.header_proto_c.append('#' + node.cpp_if)
            self.header_proto_c.append(self.continued_line(
                '', ';', 1,
                fmt_func.C_return_type, ' ', fmt_func.C_name,
                proto_list))
            if node.cpp_if:
                self.header_proto_c.append('#endif')

            impl = self.impl
            impl.append('')
            if options.debug:
                impl.append('// %s' % node.declgen)
                impl.append('// function_index=%d' % node._function_index)
            if options.doxygen and node.doxygen:
                self.write_doxygen(impl, node.doxygen)
            if node.cpp_if:
                self.impl.append('#' + node.cpp_if)
            impl.append(self.continued_line(
                '', '', 1,
                fmt_func.C_return_type, ' ', fmt_func.C_name,
                proto_list))
            impl.append('{')
            self._create_splicer(fmt_func.underscore_name +
                                 fmt_func.function_suffix, impl, C_code)
            impl.append('}')
            if node.cpp_if:
                self.impl.append('#endif  // ' + node.cpp_if)
        else:
            # There is no C wrapper, have Fortran call the function directly.
            fmt_func.C_name = node.ast.name
