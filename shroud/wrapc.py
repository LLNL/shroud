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
        self.cont = ''
        self.linelen = newlibrary.options.C_line_length
        self.doxygen_begin = '/**'
        self.doxygen_cont = ' *'
        self.doxygen_end = ' */'
        self.shared_helper = {}
        self.shared_proto_c = []

    _default_buf_args = ['arg']

    def _begin_output_file(self):
        """Start a new class for output"""
#        # forward declarations of C++ class as opaque C struct.
#        self.header_forward = {}
        # include files required by typedefs
        self.header_typedef_nodes = {}  # [arg_typedef.name] = arg_typedef
        # headers needed by implementation, i.e. helper functions
        self.header_impl_include = {} # header files in implementation file
        self.header_proto_c = []
        self.impl = []
        self.enum_impl = []
        self.struct_impl = []
        self.c_helper = {}
        self.c_helper_include = {}  # include files in generated C header

    def wrap_library(self):
        newlibrary = self.newlibrary
        fmt_library = newlibrary.fmtdict
        structs = []
        # reserved the 0 slot of capsule_order
        self.add_capsule_helper('--none--', None, [ '// Nothing to delete' ])

        self._push_splicer('class')
        for node in newlibrary.classes:
            if not node.options.wrap_c:
                continue
            if node.as_struct:
                structs.append(node)
                continue
            self._push_splicer(node.name)
            self.write_file(newlibrary, node, None)
            self._pop_splicer(node.name)
        self._pop_splicer('class')

        self.write_file(newlibrary, None, structs)

        self.write_shared_helper()

    def write_file(self, library, cls, structs):
        """Write a file for the library or class.
        """
        node = cls or library
        fmt = node.fmtdict
        self._begin_output_file()

        if structs:
            for struct in structs:
                self.wrap_struct(struct)

        if cls:
            if not cls.as_struct:
                self.wrap_class(cls)
        else:
            self.wrap_enums(library)
            self.wrap_functions(library)
            self.write_capsule_helper(library)

        c_header = fmt.C_header_filename
        c_impl = fmt.C_impl_filename

        self.gather_helper_code(self.c_helper)
        # always include helper header
        self.c_helper_include[library.fmtdict.C_header_helper] = True
        self.shared_helper.update(self.c_helper)  # accumulate all helpers

        if not self.write_header(library, cls, c_header):
            # The header will not be written if it is empty
            c_header = None
        self.write_impl(library, cls, c_header, c_impl)

    def wrap_enums(self, node):
        """Wrap all enums in a splicer block"""
        self._push_splicer('enum')
        for node in node.enums:
            self.wrap_enum(None, node)
        self._pop_splicer('enum')

    def wrap_functions(self, library):
        # worker function for write_file
        self._push_splicer('function')
        for node in library.functions:
            self.wrap_function(None, node)
        self._pop_splicer('function')

    def _gather_helper_code(self, name, done):
        """Add code from helpers.

        First recursively process dependent_helpers
        to add code in order.
        """
        if name in done:
            return  # avoid recursion
        done[name] = True

        helper_info = whelpers.CHelpers[name]
        if 'dependent_helpers' in helper_info:
            for dep in helper_info['dependent_helpers']:
                # check for recursion
                self._gather_helper_code(dep, done)

        if self.language == 'c':
            lang_header = 'c_header'
            lang_source = 'c_source'
        else:
            lang_header = 'cxx_header'
            lang_source = 'cxx_source'

        if lang_header in helper_info:
            for include in helper_info[lang_header].split():
                self.header_impl_include[include] = True
        if lang_source in helper_info:
            self.helper_source.append(helper_info[lang_source])
        elif 'source' in helper_info:
            self.helper_source.append(helper_info['source'])

        # header code using with C API  (like structs and typedefs)
        if 'h_header' in helper_info:
            for include in helper_info['h_header'].split():
                self.c_helper_include[include] = True
        if 'h_source' in helper_info:
            self.helper_header.append(helper_info['h_source'])
        if 'h_shared' in helper_info:
            self.helper_shared.append(helper_info['h_shared'])
 
    def gather_helper_code(self, helpers):
        """Gather up all helpers requested and insert code into output.

        helpers should be self.c_helper or self.shared_helper
        """
        # per class
        self.helper_source = []
        self.helper_header = []
        self.helper_shared = []

        done = {}  # avoid duplicates
        for name in sorted(helpers.keys()):
            self._gather_helper_code(name, done)

    def write_shared_helper(self):
        """Write a helper file with type definitions.
        """
        self.gather_helper_code(self.shared_helper)
        
        fmt = self.newlibrary.fmtdict
        fname = fmt.C_header_helper
        output = []

        guard = fname.replace(".", "_").upper()
        
        output.extend([
                '// For C users and %s implementation' % self.language.upper(),
                '',
                '#ifndef %s' % guard,
                '#define %s' % guard,
                ])

        if self.language == 'c++':
            output.append('')
#            if self._create_splicer('CXX_declarations', output):
#                write_file = True
            output.extend([
                    '',
                    '#ifdef __cplusplus',
                    'extern "C" {',
                    '#endif'
                    ])

        output.extend(self.helper_shared)

        if self.shared_proto_c:
            output.extend(self.shared_proto_c)

        if self.language == 'c++':
            output.extend([
                    '',
                    '#ifdef __cplusplus',
                    '}',
                    '#endif'
                    ])

        output.extend([
                '',
                '#endif  // ' + guard
                ])

        self.config.cfiles.append(
            os.path.join(self.config.c_fortran_dir, fname))
        self.write_output_file(fname, self.config.c_fortran_dir, output)

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

        # headers required by typedefs and helpers
        self.write_headers_nodes('c_header', self.header_typedef_nodes,
                                 self.c_helper_include.keys(), output)

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

        if self.helper_header:
            write_file = True
            output.extend(self.helper_header)

        if self.enum_impl:
            write_file = True
            output.extend(self.enum_impl)

        if self.struct_impl:
            write_file = True
            output.extend(self.struct_impl)

#        if self.header_forward:
#            output.extend([
#                '',
#                '// declaration of shadow types'
#            ])
#            for name in sorted(self.header_forward.keys()):
#                write_file = True
#                output.append(
#                    'struct s_{C_type_name} {{+\n'
#                    'void *addr;   /* address of C++ memory */\n'
#                    'int idtor;    /* index of destructor */\n'
#                    'int refcount; /* reference count */\n'
#                    '-}};\n'
#                    'typedef struct s_{C_type_name} {C_type_name};'.
#                    format(C_type_name=name))
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
        return write_file

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

        if hname:
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

        if self.language == 'c++':
            output.append('')
            if self._create_splicer('CXX_definitions', output):
                write_file = True
            output.append('\nextern "C" {')
        output.append('')

        if self.helper_source:
            write_file = True
            output.extend(self.helper_source)

        if self._create_splicer('C_definitions', output):
            write_file = True
        if self.impl:
            write_file = True
            output.extend(self.impl)

        if self.language == 'c++':
            output.append('')
            output.append('}  // extern "C"')

        if cls and cls.cpp_if:
            output.append('#endif  // ' + node.cpp_if)

        if write_file:
            self.config.cfiles.append(
                os.path.join(self.config.c_fortran_dir, fname))
            self.write_output_file(fname, self.config.c_fortran_dir, output)

    def wrap_struct(self, node):
        """Create a C copy of struct.
        A C++ struct must all POD.
        XXX - Only need to wrap if in a namespace.
        XXX - no need to wrap C structs
        """
        self.log.write("class {1.name}\n".format(self, node))
        typedef = node.typemap
        cname = typedef.c_type

        output = self.struct_impl
        output.append('')
        output.extend([
            '',
            'struct s_{C_type_name} {{'.format(C_type_name=cname),
            1,
        ])
        for var in node.variables:
            ast = var.ast
            result_type = ast.typename
            output.append(ast.gen_arg_as_c() + ';')
        output.extend([
            -1,
            '};',
            'typedef struct s_{C_type_name} {C_type_name};'.format(C_type_name=cname),
        ])

        # Add a sanity check on sizes of structs
        if False:
            # XXX - add this to compiled code somewhere
            typedef = node.typemap
            output.extend([
                '',
                '0#if sizeof {} != sizeof {}'.format(typedef.name, typedef.c_type),
                '0#error Sizeof {} and {} do not match'.format(typedef.name, typedef.c_type),
                '0#endif',
            ])

    def wrap_class(self, node):
        self.log.write("class {1.name}\n".format(self, node))
        cname = node.typemap.c_type

        fmt_class = node.fmtdict
        # call method syntax
        fmt_class.CXX_this_call = fmt_class.CXX_this + '->'

        # create a forward declaration for this type
        hname = whelpers.add_shadow_helper(node)
        self.shared_helper[hname] = True
#        self.header_forward[cname] = True
        self.compute_idtor(node)

        self.wrap_enums(node)

        self._push_splicer('method')
        for method in node.functions:
            self.wrap_function(node, method)
        self._pop_splicer('method')

    def compute_idtor(self, node):
        """Create a capsule destructor for type.

        Only call add_capsule_helper if the destructor is wrapped.
        Otherwise, there is no way to delete the object.
        i.e. the class has a private destructor.
        """
        has_dtor = False
        for method in node.functions:
            if '_destructor' in method.ast.attrs:
                has_dtor = True
                break

        typedef = node.typemap
        if has_dtor:
            cxx_type = typedef.cxx_type
            cxx_type = cxx_type.replace('\t', '')
            del_lines=[
                '{cxx_type} *cxx_ptr = \treinterpret_cast<{cxx_type} *>(ptr);'.format(
                    cxx_type=cxx_type) ,
                'delete cxx_ptr;',
            ]
            typedef.idtor = self.add_capsule_helper(cxx_type, typedef, del_lines)
        else:
            typedef.idtor = '0'

    def wrap_enum(self, cls, node):
        """Wrap an enumeration.
        This largly echo the C++ code
        For classes, it adds prefixes.
        """
        options = node.options
        ast = node.ast
        output = self.enum_impl

        node.eval_template('C_enum')
        fmt_enum = node.fmtdict
        fmtmembers = node._fmtmembers

        output.append('')
        append_format(output, '//  {enum_name}', fmt_enum)
        append_format(output, 'enum {C_enum} {{+', fmt_enum)
        for member in ast.members:
            fmt_id = fmtmembers[member.name]
            fmt_id.C_enum_member = wformat(options.C_enum_member_template, fmt_id)
            if member.value is not None:
                append_format(output, '{C_enum_member} = {cxx_value},', fmt_id)
            else:
                append_format(output, '{C_enum_member},', fmt_id)
        output[-1] = output[-1][:-1]        # Avoid trailing comma for older compilers
        append_format(output, '-}};', fmt_enum)

    def add_c_statements_headers(self, intent_blk):
        """Add headers required by intent_blk.
        """
        # include any dependent header in generated source
        if self.language == 'c':
            headers = intent_blk.get('c_header', None)
        else:
            headers = intent_blk.get('cxx_header', None)
        if headers:
            for h in headers.split():
                self.header_impl_include[h] = True

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
        CXX_ast = CXX_node.ast
        CXX_subprogram = CXX_node.CXX_subprogram

        # C return type
        ast = node.ast
        result_type = node.CXX_return_type
        C_subprogram = node.C_subprogram
        result_typedef = node.CXX_result_typedef
        generated_suffix = node.generated_suffix

        result_is_const = ast.const
        is_ctor = CXX_ast.attrs.get('_constructor', False)
        is_dtor = CXX_ast.attrs.get('_destructor', False)
        is_static = False
        is_allocatable = CXX_ast.attrs.get('allocatable', False)
        is_pointer = CXX_ast.is_pointer()
        is_const = ast.func_const
        is_shadow_scalar = False
        is_union_scalar = False

        if result_typedef.c_header:
            # include any dependent header in generated header
            self.header_typedef_nodes[result_typedef.name] = result_typedef
        if result_typedef.cxx_header:
            # include any dependent header in generated source
            self.header_impl_include[result_typedef.cxx_header] = True
#        if result_typedef.forward:
#            # create forward references for other types being wrapped
#            # i.e. This method returns a wrapped type
#            self.header_forward[result_typedef.c_type] = True

        if result_is_const:
            fmt_func.c_const = 'const '
        else:
            fmt_func.c_const = ''

        if CXX_subprogram == 'subroutine':
            fmt_result = fmt_func
            fmt_pattern = fmt_func
        else:
            fmt_result0 = node._fmtresult
            fmt_result = fmt_result0.setdefault('fmtc', util.Scope(fmt_func))
            fmt_result.idtor = '0'  # no destructor
            if result_typedef.c_union and not is_pointer:
                # 'convert' via fields of a union
                # used with structs where casting will not work
                # XXX - maybe change to convert to pointer to C++ struct.
                is_union_scalar = True
                fmt_result.c_var = fmt_result.C_local + fmt_result.C_result
                fmt_result.cxx_var = fmt_result.c_var
            elif result_typedef.cxx_to_c is None:
                # C and C++ are compatible
                fmt_result.c_var = fmt_result.C_local + fmt_result.C_result
                fmt_result.cxx_var = fmt_result.c_var
            else:
                fmt_result.c_var = fmt_result.C_local + fmt_result.C_result
                fmt_result.cxx_var = fmt_result.CXX_local + fmt_result.C_result

            if result_typedef.base == 'shadow' and not CXX_ast.is_indirect() and not is_ctor:
                #- decl: Class1 getClassNew() 
                is_shadow_scalar = True
                fmt_func.cxx_rv_decl = CXX_ast.gen_arg_as_cxx(
                    name=fmt_result.cxx_var, params=None, continuation=True, force_ptr=True)
            elif is_union_scalar:
                fmt_func.cxx_rv_decl = result_typedef.c_union + ' ' + fmt_result.cxx_var
            else:
                fmt_func.cxx_rv_decl = CXX_ast.gen_arg_as_cxx(
                    name=fmt_result.cxx_var, params=None, continuation=True)

            if is_ctor or is_pointer:
                # The C wrapper always creates a pointer to the new in the ctor
                fmt_result.cxx_member = '->'
                fmt_result.cxx_addr = ''
            else:
                fmt_result.cxx_member = '.'
                fmt_result.cxx_addr = '&'
            fmt_pattern = fmt_result

        proto_list = []  # arguments for wrapper prototype
        call_list = []   # arguments to call function

        # indicate which argument contains function result, usually none
        result_arg = None
        pre_call = []      # list of temporary variable declarations
        call_code = []
        post_call = []
        return_code = []

        if cls:
            # Add 'this' argument
            need_wrapper = True
            is_static = 'static' in ast.storage
            if is_ctor:
                pass
            else:
                if is_const:
                    fmt_func.c_const = 'const '
                else:
                    fmt_func.c_const = ''
                fmt_func.c_deref = '*'
                fmt_func.c_member = '->'
                fmt_func.c_var = fmt_func.C_this
                if is_static:
                    fmt_func.CXX_this_call = fmt_func.namespace_scope + fmt_func.class_scope
                else:
                    # 'this' argument
                    rvast = declast.create_this_arg(fmt_func.C_this, cls.typemap_name, is_const)
                    arg = rvast.gen_arg_as_c(continuation=True)
                    proto_list.append(arg)

                    # destructor does not need cxx_var since it passes c_var
                    # to C_memory_dtor_function (to account for reference count)
                    if not is_dtor:
                        # LHS is class' cxx_to_c
                        cls_typemap = cls.typemap
                        if cls_typemap.c_to_cxx is None:
                            # This should be set in typemap.fill_shadow_typemap_defaults
                            raise RuntimeError("Wappped class does not have c_to_cxx set")
                        append_format(
                            pre_call, 
                            '{c_const}{namespace_scope}{cxx_class} *{CXX_this} = ' +
                            cls_typemap.c_to_cxx + ';', fmt_func)

        if is_shadow_scalar:
            # Allocate a new instance, then assign pointer to dereferenced cxx_var.
            append_format(pre_call,
                          '{cxx_rv_decl} = new %s;' % result_typedef.cxx_type,
                          fmt_func)
            fmt_result.cxx_addr = ''
            fmt_result.idtor = result_typedef.idtor
            fmt_func.cxx_rv_decl = '*' + fmt_result.cxx_var

#    c_var      - argument to C function  (wrapper function)
#    c_var_trim - variable with trimmed length of c_var
#    c_var_len  - variable with length of c_var
#    cxx_var    - argument to C++ function  (wrapped function).
#                 Usually same as c_var but may be a new local variable
#                 or the function result variable.

        for arg in ast.params:
            arg_name = arg.name
            fmt_arg0 = fmtargs.setdefault(arg_name, {})
            fmt_arg = fmt_arg0.setdefault('fmtc', util.Scope(fmt_func))
            c_attrs = arg.attrs

            arg_typedef = typemap.lookup_type(arg.typename)  # XXX - look up vector
            fmt_arg.update(arg_typedef.format)

            if arg_typedef.base == 'vector':
                fmt_arg.cxx_T = c_attrs['template']

            arg_typedef, c_statements = typemap.lookup_c_statements(arg)

            fmt_arg.c_var = arg_name

            if arg.const:
                fmt_arg.c_const = 'const '
            else:
                fmt_arg.c_const = ''
            arg_is_union_scalar = False
            if arg.is_indirect():    # is_pointer?
                fmt_arg.c_deref = '*'
                fmt_arg.c_member = '->'
                fmt_arg.cxx_member = '->'
                fmt_arg.cxx_addr = ''
            else:
                fmt_arg.c_deref = ''
                fmt_arg.c_member = '.'
                fmt_arg.cxx_member = '.'
                fmt_arg.cxx_addr = '&'
                if arg_typedef.c_union:
                    arg_is_union_scalar = True
            fmt_arg.cxx_type = arg_typedef.cxx_type
            cxx_local_var = ''

            if c_attrs.get('_is_result', False):
                # This argument is the C function result
                arg_call = False

                # Note that result_type is void, so use arg_typedef.
                if arg_typedef.cxx_to_c is None:
                    fmt_arg.cxx_var = fmt_func.C_local + fmt_func.C_result
                else:
                    fmt_arg.cxx_var = fmt_func.CXX_local + fmt_func.C_result
                # Set cxx_var for C_finalize which evalutes in fmt_result context
                fmt_result.cxx_var = fmt_arg.cxx_var
                fmt_func.cxx_rv_decl = CXX_ast.gen_arg_as_cxx(
                    name=fmt_arg.cxx_var, params=None, continuation=True)

                fmt_pattern = fmt_arg
                result_arg = arg
                stmts = 'result' + generated_suffix
                need_wrapper = True
                if is_pointer:
                    fmt_arg.cxx_member = '->'
                    fmt_arg.cxx_addr = ''
                else:
                    fmt_arg.cxx_member = '.'
                    fmt_arg.cxx_addr = '&'

                if is_allocatable:
                    if not CXX_ast.is_indirect():
                        # An intermediate string * is allocated
                        # to save std::string result.
                        fmt_arg.cxx_addr = ''
                        fmt_arg.cxx_member = '->'
            else:
                arg_call = arg
                if arg_is_union_scalar:
                    # Argument is passed from Fortran to C by value.
                    # Take address of argument and pass to C++.
                    # It is dereferenced when passed to C++ to pass the value.
                    #  tutorial::struct1 * SHCXX_arg = 
                    #    static_cast<tutorial::struct1 *>(static_cast<void *>(&arg));

                    tmp = fmt_arg.c_var
                    fmt_arg.cxx_var = fmt_arg.CXX_local + fmt_arg.c_var
                    fmt_arg.c_var = '&' + tmp
                    fmt_arg.cxx_val = wformat(arg_typedef.c_to_cxx, fmt_arg)
                    fmt_arg.c_var = tmp
                    fmt_arg.cxx_decl = arg.gen_arg_as_cxx(
                        name=fmt_arg.cxx_var, params=None,
                        as_ptr=True, force_ptr=True, continuation=True)
                    append_format(pre_call, '{cxx_decl} = {cxx_val};', fmt_arg)
                elif arg_typedef.c_to_cxx is None:
                    fmt_arg.cxx_var = fmt_arg.c_var      # compatible
                else:
                    # convert C argument to C++
                    fmt_arg.cxx_var = fmt_arg.CXX_local + fmt_arg.c_var
                    fmt_arg.cxx_val = wformat(arg_typedef.c_to_cxx, fmt_arg)
                    fmt_arg.cxx_decl = arg.gen_arg_as_cxx(
                        name=fmt_arg.cxx_var, params=None, as_ptr=True, continuation=True)
                    append_format(pre_call, '{cxx_decl} = {cxx_val};', fmt_arg)

                    if arg.is_indirect():
                        # Only pointers can be passed in and must cast to another pointer.
                        # By setting cxx_local_var=pointer, it will be dereferenced
                        # correctly when passed to C++.
                        # base==string will have a pre_call block which sets cxx_local_var
                        cxx_local_var = 'pointer'
                    
                stmts = 'intent_' + c_attrs['intent'] + c_attrs.get('_generated_suffix','')

            intent_blk = c_statements.get(stmts, {})

            # Add implied buffer arguments to prototype
            for buf_arg in intent_blk.get('buf_args', self._default_buf_args):
                if buf_arg == 'arg':
                    # vector<int> -> int *
                    proto_list.append(arg.gen_arg_as_c(continuation=True))
                    continue
                elif buf_arg == 'shadow':
                    proto_list.append(arg.gen_arg_as_c(continuation=True))
                    continue

                need_wrapper = True
                if buf_arg == 'size':
                    fmt_arg.c_var_size = c_attrs['size']
                    append_format(proto_list, 'long {c_var_size}', fmt_arg)
                elif buf_arg == 'capsule':
                    fmt_arg.c_var_capsule = c_attrs['capsule']
                    append_format(proto_list, '{C_capsule_data_type} *{c_var_capsule}', fmt_arg)
                elif buf_arg == 'context':
                    fmt_arg.c_var_context = c_attrs['context']
                    append_format(proto_list, '{C_context_type} *{c_var_context}', fmt_arg)
                elif buf_arg == 'len_trim':
                    fmt_arg.c_var_trim = c_attrs['len_trim']
                    append_format(proto_list, 'int {c_var_trim}', fmt_arg)
                elif buf_arg == 'len':
                    fmt_arg.c_var_len = c_attrs['len']
                    append_format(proto_list, 'int {c_var_len}', fmt_arg)
                else:
                    raise RuntimeError("wrap_function: unhandled case {}"
                                       .format(buf_arg))

            # Add any code needed for intent(IN).
            # Usually to convert types.
            # For example, convert char * to std::string
            # Skip input arguments generated by F_string_result_as_arg
            if 'cxx_local_var' in intent_blk:
                cxx_local_var = intent_blk['cxx_local_var']
                fmt_arg.cxx_var = fmt_arg.C_argument + fmt_arg.c_var
#                    fmt_arg.cxx_var = fmt_arg.CXX_local + fmt_arg.c_var
# This uses C_local or CXX_local for arguments.
#                if 'cxx_T' in fmt_arg:
#                    fmt_arg.cxx_var = fmt_func.CXX_local + fmt_arg.c_var
#                elif arg_typedef.cxx_to_c is None:
#                    fmt_arg.cxx_var = fmt_func.C_local + fmt_arg.c_var
#                else:
#                    fmt_arg.cxx_var = fmt_func.CXX_local + fmt_arg.c_var
            if cxx_local_var == 'scalar':
                fmt_arg.cxx_member = '.'
            elif cxx_local_var == 'pointer':
                fmt_arg.cxx_member = '->'

            if self.language == 'c':
                pass
            elif arg.const:
                # cast away constness
                fmt_arg.cxx_type = arg_typedef.cxx_type
                fmt_arg.cxx_cast_to_void_ptr = wformat(
                    'static_cast<void *>\t(const_cast<'
                    '{cxx_type} *>\t({cxx_addr}{cxx_var}))', fmt_arg)
            else:
                fmt_arg.cxx_cast_to_void_ptr = wformat(
                    'static_cast<void *>({cxx_addr}{cxx_var})', fmt_arg)

            destructor_name = intent_blk.get('destructor_name', None)
            if destructor_name:
                destructor_name = wformat(destructor_name, fmt_arg)
                if destructor_name not in self.capsule_helpers:
                    del_lines = []
                    util.append_format_cmds(del_lines, intent_blk, 'destructor', fmt_arg)
                    fmt_arg.idtor = self.add_capsule_helper(destructor_name, arg_typedef, del_lines)

            # Add code for intent of argument
            # pre_call.append('// intent=%s' % intent)
            if util.append_format_cmds(pre_call, intent_blk, 'pre_call', fmt_arg):
                need_wrapper = True
            if util.append_format_cmds(post_call, intent_blk, 'post_call', fmt_arg):
                need_wrapper = True
            if 'c_helper' in intent_blk:
                c_helper = wformat(intent_blk['c_helper'], fmt_arg)
                for helper in c_helper.split():
                    self.c_helper[helper] = True

            self.add_c_statements_headers(intent_blk)

            if arg_call:
                # Collect arguments to pass to wrapped function.
                # Skips result_as_arg argument.
                if arg_is_union_scalar:
                    # Pass by value
                    call_list.append('*' + fmt_arg.cxx_var)
                elif cxx_local_var == 'scalar':
                    if arg.is_pointer():
                        call_list.append('&' + fmt_arg.cxx_var)
                    else:
                        call_list.append(fmt_arg.cxx_var)
                elif cxx_local_var == 'pointer':
                    if arg.is_pointer():
                        call_list.append(fmt_arg.cxx_var)
                    else:
                        call_list.append('*' + fmt_arg.cxx_var)
                elif arg.is_reference():
                    # reference to scalar  i.e. double &max
                    call_list.append('*' + fmt_arg.cxx_var)
                else:
                    call_list.append(fmt_arg.cxx_var)

            if arg_typedef.c_header:
                # include any dependent header in generated header
                self.header_typedef_nodes[arg_typedef.name] = arg_typedef
            if arg_typedef.cxx_header:
                # include any dependent header in generated source
                self.header_impl_include[arg_typedef.cxx_header] = True
#            if arg_typedef.forward:
#                # create forward references for other types being wrapped
#                # i.e. This argument is another wrapped type
#                self.header_forward[arg_typedef.c_type] = True
        fmt_func.C_call_list = ',\t '.join(call_list)

        fmt_func.C_prototype = options.get('C_prototype', ',\t '.join(proto_list))

        if node.return_this:
            fmt_func.C_return_type = 'void'
        elif is_dtor:
            fmt_func.C_return_type = 'void'
        elif result_typedef.base == 'shadow':
            # Return pointer to capsule_data. It contains pointer to results.
            fmt_func.C_return_type = result_typedef.c_type + ' *'
        elif fmt_func.C_custom_return_type:
            pass # fmt_func.C_return_type = fmt_func.C_return_type
        elif node.return_pointer_as == 'scalar':
            fmt_func.C_return_type = ast.gen_arg_as_c(
                name=None, as_scalar=True, params=None, continuation=True)
        else:
            fmt_func.C_return_type = ast.gen_arg_as_c(
                name=None, params=None, continuation=True)

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
            # Always create a pointer to the instance.
            fmt_func.cxx_rv_decl = result_typedef.cxx_type + ' *' + fmt_result.cxx_var
            append_format(call_code, '{cxx_rv_decl} = new {namespace_scope}'
                          '{cxx_class}({C_call_list});', fmt_func)
            if result_typedef.cxx_to_c is not None:
                fmt_func.c_rv_decl = result_typedef.c_type + ' *' + fmt_result.c_var
                fmt_result.c_val = wformat(result_typedef.cxx_to_c, fmt_result)
            fmt_result.c_type = result_typedef.c_type;
            fmt_result.idtor  = '0'
            self.header_impl_include['<stdlib.h>'] = True  # for malloc
            # XXX - similar to c_statements.result
            append_format(post_call,
                          '{c_type} *{c_var} = ({c_type} *) malloc(sizeof({c_type}));\n'
                          '{c_var}->addr = {c_val};\n'
                          '{c_var}->idtor = {idtor};\n'
                          '{c_var}->refcount = 1;',
                          fmt_result)
            C_return_code = wformat('return {c_var};', fmt_result)
        elif is_dtor:
            # Call C_memory_dtor_function to decrement reference count.
            append_format(call_code,
                          '{C_memory_dtor_function}\t(reinterpret_cast<{C_capsule_data_type} *>({C_this}), true);',
                          fmt_func)
        elif CXX_subprogram == 'subroutine':
            append_format(call_code, '{CXX_this_call}{function_name}'
                          '{CXX_template}(\t{C_call_list});', fmt_func)
        else:
            added_call_code = False

            if result_arg is None:
                # Return result from function
                # (It was not passed back in an argument)
                if self.language == 'c':
                    pass
                elif result_typedef.base == 'shadow':
                    # c_statements.post_call creates return value
                    if result_is_const:
                        # cast away constness
                        fmt_result.cxx_type = result_typedef.cxx_type
                        fmt_result.cxx_cast_to_void_ptr = wformat(
                            'static_cast<void *>\t(const_cast<'
                            '{cxx_type} *>\t({cxx_addr}{cxx_var}))', fmt_result)
                    else:
                        fmt_result.cxx_cast_to_void_ptr = wformat(
                            'static_cast<void *>({cxx_addr}{cxx_var})', fmt_result)
                elif is_union_scalar:
                    pass
                elif result_typedef.cxx_to_c is not None:
                    # Make intermediate c_var value if a conversion
                    # is required i.e. not the same as cxx_var.
                    fmt_result.c_rv_decl = CXX_ast.gen_arg_as_c(
                        name=fmt_result.c_var, params=None, continuation=True)
                    fmt_result.c_val = wformat(result_typedef.cxx_to_c, fmt_result)
                    append_format(post_call, '{c_rv_decl} = {c_val};', fmt_result)

                c_statements = result_typedef.c_statements
                generated_suffix = ast.attrs.get('_generated_suffix','')
                intent_blk = c_statements.get('result' + generated_suffix, {})
                self.add_c_statements_headers(intent_blk)

                if util.append_format_cmds(pre_call, intent_blk, 'pre_call', fmt_result):
                    need_wrapper = True
                if util.append_format_cmds(call_code, intent_blk, 'call', fmt_result):
                    need_wrapper = True
                    added_call_code = True
                if util.append_format_cmds(post_call, intent_blk, 'post_call', fmt_result):
                    need_wrapper = True
                # XXX release rv if necessary
                if 'c_helper' in intent_blk:
                    for helper in intent_blk['c_helper'].split():
                        self.c_helper[helper] = True
            elif is_allocatable:
                if not CXX_node.ast.is_indirect():
                    # Allocate intermediate string before calling function
                    fmt_arg = fmtargs[result_arg.name]['fmtc']
                    append_format(call_code, # no const
                                  'std::string * {cxx_var} = new std::string;\n'
                                  '*{cxx_var} = {CXX_this_call}{function_name}{CXX_template}'
                                  '(\t{C_call_list});', fmt_arg)
                    added_call_code = True

            if not added_call_code:
                if is_union_scalar:
                    # Call function within {}'s to assign to first field of union.
                    append_format(call_code, '{cxx_rv_decl} =\t {{{CXX_this_call}{function_name}'
                              '{CXX_template}(\t{C_call_list})}};', fmt_func)
                else:
                    append_format(call_code, '{cxx_rv_decl} =\t {CXX_this_call}{function_name}'
                              '{CXX_template}(\t{C_call_list});', fmt_func)


            if C_subprogram == 'function':
                # Note: A C function may be converted into a Fortran subroutine
                # subprogram when the result is returned in an argument.
                C_return_code = wformat('return {c_var};', fmt_result)

        if fmt_func.inlocal('C_finalize' + generated_suffix):
            # maybe check C_finalize up chain for accumulative code
            # i.e. per class, per library.
            finalize_line = fmt_func.get('C_finalize' + generated_suffix)
            need_wrapper = True
            post_call.append('{')
            post_call.append('    // C_finalize')
            util.append_format_indent(post_call, finalize_line, fmt_result)
            post_call.append('}')

        if fmt_func.inlocal('C_return_code'):
            need_wrapper = True
            C_return_code = wformat(fmt_func.C_return_code, fmt_func)
        elif is_union_scalar:
            fmt_func.C_return_code = wformat('return {cxx_var}.c;', fmt_result)
        elif node.return_pointer_as == 'scalar':
            # dereference pointer to return scalar
            fmt_func.C_return_code = wformat('return *{cxx_var};', fmt_result)
        else:
            fmt_func.C_return_code = C_return_code

        if pre_call:
            fmt_func.C_pre_call = '\n'.join(pre_call)
        fmt_func.C_call_code = '\n'.join(call_code)
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
            C_code.extend(call_code)
            C_code.extend(post_call_pattern)
            C_code.extend(post_call)
            C_code.append(fmt_func.C_return_code)
            C_code.append(-1)

        if need_wrapper:
            self.header_proto_c.append('')
            if node.cpp_if:
                self.header_proto_c.append('#' + node.cpp_if)
            append_format(self.header_proto_c,
                          '{C_return_type} {C_name}(\t{C_prototype});',
                          fmt_func)
            if node.cpp_if:
                self.header_proto_c.append('#endif')

            impl = self.impl
            impl.append('')
            if options.debug:
                impl.append('// ' + node.declgen)
                impl.append('// function_index=%d' % node._function_index)
            if options.doxygen and node.doxygen:
                self.write_doxygen(impl, node.doxygen)
            if node.cpp_if:
                self.impl.append('#' + node.cpp_if)
            append_format(impl,
                          '{C_return_type} {C_name}(\t{C_prototype})',
                          fmt_func)
            impl.append('{')
            self._create_splicer(fmt_func.underscore_name +
                                 fmt_func.function_suffix, impl, C_code)
            impl.append('}')
            if node.cpp_if:
                impl.append('#endif  // ' + node.cpp_if)
        else:
            # There is no C wrapper, have Fortran call the function directly.
            fmt_func.C_name = node.ast.name

    def write_capsule_helper(self, library):
        """Write a function used to delete memory when C/C++
        memory is deleted.
        """
        if len(self.capsule_order) == 1:
            # Only the 0 slot has been added, so return
            # since the function will be unused.
            return
        options = library.options

        self.c_helper['capsule_data_helper'] = True
        fmt = library.fmtdict

        self.header_impl_include.update(self.capsule_include)
        self.header_impl_include[fmt.C_header_helper] = True

        append_format(
            self.shared_proto_c,
            '\nvoid {C_memory_dtor_function}\t({C_capsule_data_type} *cap, bool gc);',
            fmt)

        output = self.impl
        append_format(
            output,
            '\n'
            '// Release C++ allocated memory.\n'
            'void {C_memory_dtor_function}\t({C_capsule_data_type} *cap, bool gc)\n'
            '{{+'
            , fmt)

        if options.F_auto_reference_count:
            # check refererence before deleting
            append_format(
                output,
                '@--cap->refcount;\n'
                'if (cap->refcount > 0) {{+\n'
                'return;\n'
                '-}}'
                , fmt)

        append_format(
            output,
            'void *ptr = cap->addr;\n'
            'switch (cap->idtor) {{'
            , fmt)

        for i, name in enumerate(self.capsule_order):
            output.append('case {}:\n{{+'.format(i))
            output.extend(self.capsule_helpers[name][1])
            output.append('break;\n-}')

        output.append(
            'default:\n{+\n'
            '// Unexpected case in destructor\n'
            'break;\n'
            '-}\n'
            '}\n'
            'if (gc) {+\n'
            'free(cap);\n'
            '-} else {+\n'
            'cap->addr = NULL;\n'
            'cap->idtor = 0;  // avoid deleting again\n'
            '-}\n'
            '-}'
        )
        self.header_impl_include['<stdlib.h>'] = True  # for free

    capsule_helpers = {}
    capsule_order = []
    capsule_include = {}  # includes needed by C_memory_dtor_function
    def add_capsule_helper(self, name, typemap, lines):
        """Add unique names to capsule_helpers.
        Return index of name.
        """
        if name not in self.capsule_helpers:
            self.capsule_helpers[name] = (str(len(self.capsule_helpers)), lines)
            self.capsule_order.append(name)

            # include files required by the type
            if typemap and typemap.cxx_header:
                for include in typemap.cxx_header.split():
                    self.capsule_include[include] = True

        return self.capsule_helpers[name][0]
