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

from . import whelpers
from . import util
from .util import append_format

wformat = util.wformat


class Wrapc(util.WrapperMixin):
    """Generate C bindings for C++ classes

    """
    def __init__(self, tree, config, splicers):
        self.tree = tree    # json tree
        self.patterns = tree['patterns']
        self.language = tree['language']
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

    def _c_type(self, lang, arg):
        """
        Return the C type.
        pass-by-value default

        attributes:
        ptr - True = pass-by-reference
        reference - True = pass-by-reference

        """
#        if lang not in [ 'c_type', 'cpp_type' ]:
#            raise RuntimeError
        t = []
        typedef = util.Typedef.lookup(arg['type'])
        attrs = arg['attrs']
        if 'template' in attrs:
            # If a template, use its type
            typedef = util.Typedef.lookup(attrs['template'])
        if typedef is None:
            raise RuntimeError("No such type %s" % arg['type'])
        if arg['const']:
            t.append('const')
        typ = getattr(typedef, lang)
        if typ is None:
            raise RuntimeError(
                "Type {} has no value for {}".format(arg['type'], lang))
        t.append(typ)
        if attrs.get('ptr', False):
            t.append('*')
        elif attrs.get('reference', False):
            if lang == 'cpp_type':
                t.append('&')
            else:
                t.append('*')
        return ' '.join(t)

    def _c_decl(self, lang, arg, name=None):
        """
        Return the C declaration.

        If name is not supplied, use name in arg.
        This makes it easy to reproduce the arguments.
        """
#        if lang not in [ 'c_type', 'cpp_type' ]:
#            raise RuntimeError
        typ = self._c_type(lang, arg)
        return typ + ' ' + (name or arg['name'])

    def wrap_library(self):
        fmt_library = self.tree['fmt']

        self._push_splicer('class')
        for node in self.tree['classes']:
            self._push_splicer(node['name'])
            self.write_file(self.tree, node)
            self._pop_splicer(node['name'])
        self._pop_splicer('class')

        if self.tree['functions']:
            self.write_file(self.tree, None)

    def write_file(self, library, cls):
        """Write a file for the library and its functions or
        a class and its methods.
        """
        node = cls or library
        fmt = node['fmt']
        self._begin_output_file()
        if cls:
            self.wrap_class(cls)
        else:
            self.wrap_functions(library)
        c_header = fmt.C_header_filename
        c_impl = fmt.C_impl_filename
        self.write_header(library, cls, c_header)
        self.write_impl(library, cls, c_header, c_impl)

    def wrap_functions(self, tree):
        # worker function for write_file
        self._push_splicer('function')
        for node in tree['functions']:
            self.wrap_function(None, node)
        self._pop_splicer('function')

    def write_header(self, library, cls, fname):
        """ Write header file for a library node or a class node.
        """
        guard = fname.replace(".", "_").upper()
        node = cls or library
        options = node['options']

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
        output.extend([
                '',
                '#endif  // %s' % guard
                ])

        if write_file:
            self.config.cfiles.append(
                os.path.join(self.config.c_fortran_dir, fname))
            self.write_output_file(fname, self.config.c_fortran_dir, output)

    def write_impl(self, library, cls, hname, fname):
        """Write implementation
        """
        node = cls or library
        options = node['options']

        # If no C wrappers are required, do not write the file
        write_file = False
        output = []
        output.append('// ' + fname)

        # Insert any helper functions needed
        helper_source = []
        if self.c_helper:
            helperdict = whelpers.find_all_helpers('c', self.c_helper)
            helpers = sorted(self.c_helper)
            if self.language == 'c':
                lang_header = 'c_header'
                lang_source = 'c_source'
            else:
                lang_header = 'cpp_header'
                lang_source = 'cpp_source'
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
        if cls and cls['cpp_header']:
            for include in cls['cpp_header'].split():
                self.header_impl_include[include] = True
        else:
            for include in library['cpp_header'].split():
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
        self.log.write("class {1[name]}\n".format(self, node))
        name = node['name']
        typedef = util.Typedef.lookup(name)
        cname = typedef.c_type

        fmt_class = node['fmt']
        # call method syntax
        fmt_class.CPP_this_call = fmt_class.CPP_this + '->'
#        fmt_class.update(dict(
#                ))

        # create a forward declaration for this type
        self.header_forward[cname] = True

        self._push_splicer('method')
        for method in node['methods']:
            self.wrap_function(node, method)
        self._pop_splicer('method')

    def wrap_function(self, cls, node):
        """
        Wrap a C++ function with C
        cls  - class node or None for functions
        node - function/method node
        """
        options = node['options']
        if not options.wrap_c:
            return

        if cls:
            cls_function = 'method'
        else:
            cls_function = 'function'
        self.log.write("C {0} {1[_decl]}\n".format(cls_function, node))

        fmt_func = node['fmt']

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
            lang_header = 'cpp_header'

        # Look for C++ routine to wrap
        # Usually the same node unless it is generated (i.e. bufferified)
        CPP_node = node
        generated = []
        if '_generated' in CPP_node:
            generated.append(CPP_node['_generated'])
        while '_PTR_C_CPP_index' in CPP_node:
            CPP_node = self.tree['function_index'][
                CPP_node['_PTR_C_CPP_index']]
            if '_generated' in CPP_node:
                generated.append(CPP_node['_generated'])
        CPP_result = CPP_node['result']
        CPP_result_type = CPP_result['type']
        CPP_subprogram = CPP_node['_subprogram']

        # C return type
        result = node['result']
        result_type = result['type']
        subprogram = node['_subprogram']
        generator = node.get('_generated', '')
        intent_grp = ''
        if generator == 'arg_to_buffer':
            intent_grp = '_buf'

        # C++ functions which return 'this',
        # are easier to call from Fortran if they are subroutines.
        # There is no way to chain in Fortran:  obj->doA()->doB();
        if node.get('return_this', False):
            CPP_result_type = 'void'
            CPP_subprogram = 'subroutine'

        result_typedef = util.Typedef.lookup(result_type)
        result_is_const = result['const']
        is_ctor = node['attrs'].get('constructor', False)
        is_dtor = node['attrs'].get('destructor', False)
        is_const = node['attrs'].get('const', False)

        if result_typedef.c_header:
            # include any dependent header in generated header
            self.header_typedef_include[result_typedef.c_header] = True
        if result_typedef.cpp_header:
            # include any dependent header in generated source
            self.header_impl_include[result_typedef.cpp_header] = True
        if result_typedef.forward:
            # create forward references for other types being wrapped
            # i.e. This method returns a wrapped type
            self.header_forward[result_typedef.c_type] = True

        if result_is_const:
            fmt_func.c_const = 'const '
        else:
            fmt_func.c_const = ''

        return_lang = '{cpp_var}'  # Assume C and C++ types are compatiable
        if CPP_subprogram == 'subroutine':
            fmt_result = fmt_func
            fmt_pattern = fmt_func
        else:
            fmt_result = result.setdefault('fmtc', util.Options(fmt_func))
            fmt_result.cpp_var = fmt_func.C_result
#            fmt_result.cpp_decl = self._c_type('cpp_type', CPP_result)

            fmt_result.cpp_rv_decl = self._c_decl('cpp_type', CPP_result, name=fmt_func.C_result)
            fmt_pattern = fmt_result

        proto_list = []
        call_list = []
        if cls:
            need_wrapper = True
            # object pointer
            arg_dict = dict(name=fmt_func.C_this,
                            type=cls['name'],
                            const=is_const,
                            attrs=dict(ptr=True))
            C_this_type = self._c_type('c_type', arg_dict)
            if not is_ctor:
                arg = self._c_decl('c_type', arg_dict)
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
            # LHS is class' cpp_to_c
            cls_typedef = util.Typedef.lookup(cls['name'])
            append_format(pre_call, 
                          '{c_const}{cpp_class} *{CPP_this} = ' +
                          cls_typedef.c_to_cpp + ';', fmt_func)

#    c_var      - argument to C function  (wrapper function)
#    c_var_trim - variable with trimmed length of c_var
#    c_var_len  - variable with length of c_var
#    cpp_var    - argument to C++ function  (wrapped function).
#                 Usually same as c_var but may be a new local variable
#                 or the funtion result variable.

        for arg in node['args']:
            fmt_arg = arg.setdefault('fmtc', util.Options(fmt_func))
            c_attrs = arg['attrs']
            arg_typedef, c_statements = util.lookup_c_statements(arg)
            if 'template' in c_attrs:
                fmt_arg.cpp_T = c_attrs['template']

            fmt_arg.c_var = arg['name']

            if arg['const']:
                fmt_arg.c_const = 'const '
            else:
                fmt_arg.c_const = ''
            if c_attrs.get('ptr', False):
                fmt_arg.c_ptr = ' *'
            else:
                fmt_arg.c_ptr = ''
            fmt_arg.cpp_type = arg_typedef.cpp_type

            proto_list.append(self._c_decl('c_type', arg))

            if c_attrs.get('_is_result', False):
                arg_call = False
                fmt_arg.cpp_var = fmt_arg.C_result
                fmt_pattern = fmt_arg
                result_arg = arg
                stmts = 'result' + intent_grp
                need_wrapper = True
            else:
                arg_call = arg
                fmt_arg.cpp_var = fmt_arg.c_var      # name in c++ call.
                stmts = 'intent_' + c_attrs['intent'] + intent_grp

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
            have_cpp_local_var = arg_typedef.cpp_local_var or \
                intent_blk.get('cpp_local_var', False)
            if have_cpp_local_var:
                fmt_arg.cpp_var = 'SH_' + fmt_arg.c_var

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
                # pick up c_str() from cpp_to_c
                fmt_arg.cpp_val = wformat(arg_typedef.cpp_to_c, fmt_arg)
                for cmd in cmd_list:
                    append_format(post_call, cmd, fmt_arg)

            if 'c_helper' in intent_blk:
                for helper in intent_blk['c_helper'].split():
                    self.c_helper[helper] = True

            cpp_header = intent_blk.get(lang_header, None)
            # include any dependent header in generated source
            if cpp_header:
                for h in cpp_header.split():
                    self.header_impl_include[h] = True

            if arg_typedef.cpp_local_var:
                # cpp_local_var should only be set if c_statements are not used
                if arg_typedef.c_statements:
                    raise RuntimeError(
                        'c_statements and cpp_local_var are both '
                        'defined for {}'
                        .format(arg_typedef.name))
                append_format(pre_call,
                              '{c_const}{cpp_type}{c_ptr} {cpp_var} = ' +
                              arg_typedef.c_to_cpp + ';', fmt_arg)

            if arg_call:
                if have_cpp_local_var:
                    call_list.append(fmt_arg.cpp_var)
                else:
                    # convert C argument to C++
                    append_format(call_list, arg_typedef.c_to_cpp, fmt_arg)

            if arg_typedef.c_header:
                # include any dependent header in generated header
                self.header_typedef_include[arg_typedef.c_header] = True
            if arg_typedef.cpp_header:
                # include any dependent header in generated source
                self.header_impl_include[arg_typedef.cpp_header] = True
            if arg_typedef.forward:
                # create forward references for other types being wrapped
                # i.e. This argument is another wrapped type
                self.header_forward[arg_typedef.c_type] = True
        fmt_func.C_call_list = ', '.join(call_list)

        fmt_func.C_prototype = options.get('C_prototype', ', '.join(proto_list))

        if node.get('return_this', False):
            fmt_func.C_return_type = 'void'
        elif 'C_return_type' in node:
            fmt_func.C_return_type = node['C_return_type']
        else:
            fmt_func.C_return_type = options.get(
                'C_return_type', self._c_type('c_type', result))

        if pre_call:
            fmt_func.C_pre_call = '\n'.join(pre_call)
        if post_call:
            fmt_func.C_post_call = '\n'.join(post_call)

        post_call_pattern = []
        if 'C_error_pattern' in node:
            C_error_pattern = node['C_error_pattern'] + \
                              node.get('_error_pattern_suffix', '')
            if C_error_pattern in self.patterns:
                post_call_pattern.append('// C_error_pattern')
                append_format(
                    post_call_pattern, self.patterns[C_error_pattern], fmt_pattern)
        if post_call_pattern:
            need_wrapper = True
            fmt_func.C_post_call_pattern = '\n'.join(post_call_pattern)

        # body of function
        splicer_code = self.splicer_stack[-1].get(fmt_func.function_name, None)
        if 'C_code' in node:
            need_wrapper = True
            C_code = [1, wformat(node['C_code'], fmt_func), -1]
        elif splicer_code:
            need_wrapper = True
            C_code = splicer_code
        else:
            # generate the C body
            fmt_func.C_return_code = 'return;'
            if is_ctor:
                fmt_func.C_call_code = wformat('{cpp_rv_decl} = new {cpp_class}'
                               '({C_call_list});', fmt_result)
                fmt_func.C_return_code = ('return '
                               + wformat(result_typedef.cpp_to_c, fmt_result)
                               + ';')
            elif is_dtor:
                fmt_func.C_call_code = 'delete %s;' % fmt_func.CPP_this
            elif CPP_subprogram == 'subroutine':
                fmt_func.C_call_code = wformat(
                    '{CPP_this_call}{function_name}'
                    '{CPP_template}({C_call_list});',
                    fmt_func)
            else:
                fmt_func.C_call_code = wformat(
                    '{cpp_rv_decl} = {CPP_this_call}{function_name}'
                    '{CPP_template}({C_call_list});',
                    fmt_result)

                if not result_arg:
                    # The result is not passed back in an argument
                    c_statements = result_typedef.c_statements
                    intent_blk = c_statements.get('result', {})
                    if result_typedef.cpp_to_c != '{cpp_var}':
                        # Make intermediate c_var value if a conversion
                        # is required i.e. not the same as cpp_var.
                        have_c_local_var = True
                    else:
                        have_c_local_var = intent_blk.get('c_local_var', False)
                    if have_c_local_var:
                        # XXX need better mangling than 'X'
                        fmt_result.c_var = 'X' + fmt_func.C_result
                        fmt_result.c_rv_decl = self._c_decl('c_type', CPP_result,
                                                          name=fmt_result.c_var)
                        fmt_result.c_val = wformat(result_typedef.cpp_to_c, fmt_result)
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
                    # Note: A C function may be converted into a Fortran subroutine subprogram
                    # when the result is returned in an argument.
                    fmt_func.C_return_code = ('return '
                                            + wformat(return_lang, fmt_result)
                                            + ';')

            if 'C_post_call' in node:
                need_wrapper = True
                post_call.append('{')
                post_call.append('// C_post_call')
                append_format(post_call, node['C_post_call'], fmt_func)
                post_call.append('}')

            if 'C_return_code' in node:
                # override any computed return code.
                need_wrapper = True
                fmt_func.C_return_code = wformat(node['C_return_code'], fmt_func)

            # copy-out values, clean up
            C_code = [1]
            C_code.extend(pre_call)
            C_code.append(fmt_func.C_call_code)
            C_code.extend(post_call_pattern)
            C_code.extend(post_call)
            C_code.append(fmt_func.C_return_code)
            C_code.append(-1)

        if need_wrapper:
            self.header_proto_c.append('')
            self.header_proto_c.append(
                wformat('{C_return_type} {C_name}({C_prototype});',
                        fmt_func))

            impl = self.impl
            impl.append('')
            if options.debug:
                impl.append('// %s' % node['_decl'])
                impl.append('// function_index=%d' % node['_function_index'])
            if options.doxygen and 'doxygen' in node:
                self.write_doxygen(impl, node['doxygen'])
            impl.append(wformat('{C_return_type} {C_name}({C_prototype})', fmt_func))
            impl.append('{')
            self._create_splicer(fmt_func.underscore_name +
                                 fmt_func.function_suffix, impl, C_code)
            impl.append('}')
        else:
            # There is no C wrapper, have Fortran call the function directly.
            fmt_func.C_name = node['result']['name']


    def XXXget_intent(self, intent_blk, block):
        # Maybe later...
        """Get a language specific block of code.
        block = pre_call, post_call

        intent_in={
            pre_call_c = []
            pre_call_cpp = []
        }
        -- or --
        intent_in={
            pre_call = []
        }
        """
        name = block + '_' + self.language
        if name in intent_blk:
            return intent_blk[name]
        elif block in intent_blk:
            return intent_blk[block]
        else:
            return []
