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
Generate Fortran bindings for C++ code.

module {F_module_name}

  type {F_derived_name}
    type(C_PTR) {F_derived_member}
  contains
    procedure :: {F_name_function} => {F_name_impl}
    generic :: {F_name_generic} => {F_name_function}, ...
  end type {F_derived_name}

  ! interface for C functions
  interface
    {F_C_pure_clause}{F_C_subprogram} {F_C_name}({F_C_arguments}) &
        {F_C_result_clause} &
        bind(C, name="{C_name}")
      {arg_c_decl}
    end {F_C_subprogram} {F_C_name}
  end interface

  interface {F_name_generic}
    module procedure {F_name_impl}
  end interface {F_name_generic}

contains

  {F_pure_clause} {F_subprogram} {F_name_impl}({F_arguments}){F_result_clause}
      {F_C_name}({F_arg_c_call})
     {F_code}
  end {F_subprogram} {F_name_impl}

end module {F_module_name}
----------
"""
from __future__ import print_function
from __future__ import absolute_import

import copy
import re
import os

from . import declast
from . import todict
from . import typemap
from . import whelpers
from . import util
from .util import wformat, append_format


class Wrapf(util.WrapperMixin):
    """Generate Fortran bindings.
    """

    def __init__(self, newlibrary, config, splicers):
        self.newlibrary = newlibrary
        self.patterns = newlibrary.patterns
        self.config = config
        self.log = config.log
        self._init_splicer(splicers)
        self.comment = '!'
        self.cont = ' &'
        self.linelen = newlibrary.options.F_line_length
        self.doxygen_begin = '!>'
        self.doxygen_cont = '!!'
        self.doxygen_end = '!<'

    def _begin_output_file(self):
        """Start a new class for output"""
        self.use_stmts = []
        self.enum_impl = []
        self.f_type_decl = []
        self.c_interface = []
        self.abstract_interface = []
        self.generic_interface = []
        self.impl = []          # implementation, after contains
        self.operator_impl = []
        self.operator_map = {}  # list of function names by operator
        # {'.eq.': [ 'abc', 'def'] }
        self.c_interface.append('')
        self.c_interface.append('interface')
        self.c_interface.append(1)
        self.f_function_generic = {}  # look for generic functions
        self.f_abstract_interface = {}
        self.f_helper = {}

    def _end_output_file(self):
        self.c_interface.append(-1)
        self.c_interface.append('end interface')

    def _begin_class(self):
        self.f_type_generic = {}  # look for generic methods
        self.type_bound_part = []

    def wrap_library(self):
        newlibrary = self.newlibrary
        options = newlibrary.options
        fmt_library = newlibrary.fmtdict
        fmt_library.F_result_clause = ''
        fmt_library.F_pure_clause = ''
        fmt_library.F_C_result_clause = ''
        fmt_library.F_C_pure_clause = ''

        self.module_use = {}    # Use statements for a module
        self._begin_output_file()
        self._push_splicer('class')
        for node in newlibrary.classes:
            if not node.options.wrap_fortran:
                continue
            self._begin_class()

            name = node.name
            # how to decide module name, module per class
#            module_name = node.options.setdefault('module_name', name.lower())
            if node.as_struct:
                self.wrap_struct(node)
            else:
                self.wrap_class(node)
            if options.F_module_per_class:
                self._pop_splicer('class')
                self._end_output_file()
                self.write_module(newlibrary, node)
                self._begin_output_file()
                self._push_splicer('class')
        self._pop_splicer('class')

        if newlibrary.functions or newlibrary.enums:
            self._begin_class()  # clear out old class info
            if options.F_module_per_class:
                self._begin_output_file()
            newlibrary.F_module_dependencies = []

            self.wrap_enums(newlibrary)

            self._push_splicer('function')
            for node in newlibrary.functions:
                self.wrap_function(None, node)
            self._pop_splicer('function')

            self.c_interface.append('')
            self._create_splicer('additional_interfaces', self.c_interface)
            self.impl.append('')
            self._create_splicer('additional_functions', self.impl)

            if options.F_module_per_class:
                # library module
                self._end_output_file()
                self._create_splicer('module_use', self.use_stmts)
                self.write_module(newlibrary, None)

        if not options.F_module_per_class:
            # put all functions and classes into one module
            self._end_output_file()
            self.write_module(newlibrary, None)

        self.write_c_helper()

    def wrap_struct(self, node):
        """A struct must be bind(C)-able. i.e. all POD.
        No methods.
        """
        self.log.write("class {1.name}\n".format(self, node))
        typedef = node.typedef

        fmt_class = node.fmtdict

        fmt_class.F_derived_name = typedef.f_derived_type

        # type declaration
        output = self.f_type_decl
        output.append('')
        self._push_splicer(fmt_class.cxx_class)
        output.extend([
                '',
                wformat('type, bind(C) :: {F_derived_name}', fmt_class),
                1,
                ])
        for var in node.variables:
            ast = var.ast
            result_type = ast.typename
            typedef = typemap.Typedef.lookup(result_type)
            output.append(ast.gen_arg_as_fortran())
            self.update_f_module(self.module_use, typedef.f_module)
        output.extend([
                 -1,
                 wformat('end type {F_derived_name}', fmt_class),
                 ])
        self._pop_splicer(fmt_class.cxx_class)

    def wrap_class(self, node):
        self.log.write("class {1.name}\n".format(self, node))
        typedef = node.typedef

        fmt_class = node.fmtdict

        fmt_class.F_derived_name = typedef.f_derived_type

        # wrap methods
        self._push_splicer(fmt_class.cxx_class)
        self._create_splicer('module_use', self.use_stmts)

        self.wrap_enums(node)

        self._push_splicer('method')
        for method in node.functions:
            self.wrap_function(node, method)
        self._pop_splicer('method')

        self.write_object_get_set(node, fmt_class)
        self.impl.append('')
        self._create_splicer('additional_functions', self.impl)
        self._pop_splicer(fmt_class.cxx_class)

        # type declaration
        self.f_type_decl.append('')
        self._push_splicer(fmt_class.cxx_class)
        self._create_splicer('module_top', self.f_type_decl)
        self.f_type_decl.extend([
                '',
                wformat('type {F_derived_name}', fmt_class),
                1,
                wformat('type(C_PTR), private :: {F_derived_member}', fmt_class),
                ])
        self.set_f_module(self.module_use, 'iso_c_binding', 'C_PTR')
        self._create_splicer('component_part', self.f_type_decl)
        self.f_type_decl.extend([
                -1, 'contains', 1,
                ])
        self.f_type_decl.extend(self.type_bound_part)

        # Look for generics
        # splicer to extend generic
#        self._push_splicer('generic')
        f_type_decl = self.f_type_decl
        for key in sorted(self.f_type_generic.keys()):
            methods = self.f_type_generic[key]
            if len(methods) > 1:

                # Look for any cpp_if declarations
                any_cpp_if = False
                for node in methods:
                    if node.cpp_if:
                        any_cpp_if = True
                        break

                if any_cpp_if:
                    # If using cpp, add a generic line for each function
                    # to avoid conditional/continuation problems.
                    for node in methods:
                        if node.cpp_if:
                            f_type_decl.append('#' + node.cpp_if)
                        f_type_decl.append('generic :: {} => {}'.format(
                            key, node.fmtdict.F_name_function))
                        if node.cpp_if:
                            f_type_decl.append('#endif')
                else:
                    parts = [ 'generic :: ', key, ' => ' ]
                    for node in methods:
                        parts.append(node.fmtdict.F_name_function)
                        parts.append(', ')
                    del parts[-1]
                    f_type_decl.append('\t'.join(parts))
#                    self._create_splicer(key, self.f_type_decl)
#        self._pop_splicer('generic')

        self._create_splicer('type_bound_procedure_part', self.f_type_decl)
        self.f_type_decl.extend([
                 -1,
                 wformat('end type {F_derived_name}', fmt_class),
                 ])

        self.c_interface.append('')
        self._create_splicer('additional_interfaces', self.c_interface)

        self._pop_splicer(fmt_class.cxx_class)

        # overload operators
        self.overload_compare(
            fmt_class, '.eq.', fmt_class.class_lower + '_eq',
            wformat('c_associated(a%{F_derived_member}, b%{F_derived_member})',
                    fmt_class))
#        self.overload_compare(fmt_class, '==', fmt_class.class_lower + '_eq', None)
        self.overload_compare(
            fmt_class, '.ne.', fmt_class.class_lower + '_ne',
            wformat(
                '.not. c_associated'
                '(a%{F_derived_member}, b%{F_derived_member})',
                fmt_class))
#        self.overload_compare(fmt_class, '/=', fmt_class.class_lower + '_ne', None)

    def wrap_enums(self, node):
        """Wrap all enums in a splicer block"""
        self._push_splicer('enum')
        for node in node.enums:
            self.wrap_enum(None, node)
        self._pop_splicer('enum')

    def wrap_enum(self, cls, node):
        """Wrap an enumeration.
        Create an integer parameter for each member.
        """
        options = node.options
        ast = node.ast
        output = self.enum_impl

        fmt_enum = node.fmtdict
        fmtmembers = node._fmtmembers

        output.append('')
        append_format(output, '!  {enum_name}', fmt_enum)
        for member in ast.members:
            fmt_id = fmtmembers[member.name]
            fmt_id.F_enum_member = wformat(options.F_enum_member_template, fmt_id)
            append_format(output, 'integer(C_INT), parameter :: {F_enum_member} = {evalue}', 
                          fmt_id)
        self.set_f_module(self.module_use, 'iso_c_binding', 'C_INT')

    def write_object_get_set(self, node, fmt_class):
        """Write get and set methods for instance pointer.

        node = class dictionary
        """
        options = node.options
        impl = self.impl
        fmt = util.Scope(fmt_class)

        # get
        fmt.underscore_name = fmt_class.F_name_instance_get
        if fmt.underscore_name:
            fmt.F_name_function = wformat(options.F_name_function_template, fmt)
            fmt.F_name_impl = wformat(options.F_name_impl_template, fmt)

            self.type_bound_part.append('procedure :: %s => %s' % (
                    fmt.F_name_function, fmt.F_name_impl))

            impl.append('')
            append_format(
                impl, 'function {F_name_impl}({F_this}) '
                'result ({F_derived_member})', fmt)
            impl.append(1)
            impl.append('use iso_c_binding, only: C_PTR')
            append_format(
                impl, 'class({F_derived_name}), intent(IN) :: {F_this}', fmt)
            append_format(impl, 'type(C_PTR) :: {F_derived_member}', fmt)
            append_format(impl, '{F_derived_member} = {F_this}%{F_derived_member}', fmt)
            impl.append(-1)
            append_format(impl, 'end function {F_name_impl}', fmt)

        # set
        fmt.underscore_name = fmt_class.F_name_instance_set
        if fmt.underscore_name:
            fmt.F_name_function = wformat(options.F_name_function_template, fmt)
            fmt.F_name_impl = wformat(options.F_name_impl_template, fmt)

            self.type_bound_part.append('procedure :: %s => %s' % (
                    fmt.F_name_function, fmt.F_name_impl))

            impl.append('')
            append_format(
                impl, 'subroutine {F_name_impl}'
                '({F_this}, {F_derived_member})', fmt)
            impl.append(1)
            impl.append('use iso_c_binding, only: C_PTR')
            append_format(
                impl, 'class({F_derived_name}), intent(INOUT) :: {F_this}',
                fmt)
            append_format(
                impl, 'type(C_PTR), intent(IN) :: {F_derived_member}', fmt)
            append_format(impl, '{F_this}%{F_derived_member} = {F_derived_member}', fmt)
            impl.append(-1)
            append_format(impl, 'end subroutine {F_name_impl}', fmt)

        # associated
        fmt.underscore_name = fmt_class.F_name_associated
        if fmt.underscore_name:
            fmt.F_name_function = wformat(options.F_name_function_template, fmt)
            fmt.F_name_impl = wformat(options.F_name_impl_template, fmt)

            self.type_bound_part.append('procedure :: %s => %s' % (
                    fmt.F_name_function, fmt.F_name_impl))

            impl.append('')
            append_format(
                impl, 'function {F_name_impl}({F_this}) result (rv)', fmt)
            impl.append(1)
            impl.append('use iso_c_binding, only: c_associated')
            append_format(
                impl, 'class({F_derived_name}), intent(IN) :: {F_this}', fmt)
            impl.append('logical rv')
            append_format(impl, 'rv = c_associated({F_this}%{F_derived_member})', fmt)
            impl.append(-1)
            append_format(impl, 'end function {F_name_impl}', fmt)

    def overload_compare(self, fmt_class, operator, procedure, predicate):
        """ Overload .eq. and .eq.
        """
        fmt = util.Scope(fmt_class)
        fmt.procedure = procedure
        fmt.predicate = predicate

        ops = self.operator_map.setdefault(operator, [])
        ops.append(procedure)

        if predicate is None:
            # .eq. and == use same function
            return

        operator = self.operator_impl
        operator.append('')
        append_format(operator, 'function {procedure}(a,b) result (rv)', fmt)
        operator.append(1)
        operator.append('use iso_c_binding, only: c_associated')
        append_format(operator,
                      'type({F_derived_name}), intent(IN) ::a,b', fmt)
        operator.append('logical :: rv')
        append_format(operator, 'if ({predicate}) then', fmt)
        operator.append(1)
        operator.append('rv = .true.')
        operator.append(-1)
        operator.append('else')
        operator.append(1)
        operator.append('rv = .false.')
        operator.append(-1)
        operator.append('endif')
        operator.append(-1)
        append_format(operator, 'end function {procedure}', fmt)

    def wrap_function(self, cls, node):
        """
        cls  - class node or None for functions
        node - function/method node

        Wrapping involves both a C interface and a Fortran wrapper.
        For some generic functions there may be single C method with
        multiple Fortran wrappers.

        """
        if cls:
            cls_function = 'method'
        else:
            cls_function = 'function'

        options = node.options
        wrap = []
        if options.wrap_c:
            wrap.append('C-interface')
        if options.wrap_fortran:
            wrap.append('Fortran')
        if not wrap:
            return

        self.log.write(', '.join(wrap))
        self.log.write(" {0} {1.declgen}\n".format(cls_function, node))

        # Create fortran wrappers first.
        # If no real work to do, call the C function directly.
        if options.wrap_fortran:
            self.wrap_function_impl(cls, node)
        if options.wrap_c:
            self.wrap_function_interface(cls, node)

    def update_f_module(self, modules, f_module):
        """aggragate the information from f_module into modules.
        modules is a dictionary of dictionaries:
            modules['iso_c_bindings']['C_INT'] = True
        f_module is a dictionary of lists:
            dict(iso_c_binding=['C_INT'])
        """
        if f_module is not None:
            for mname, only in f_module.items():
                module = modules.setdefault(mname, {})
                if only:  # Empty list means no ONLY clause
                    for oname in only:
                        module[oname] = True

    def set_f_module(self, modules, mname, *only):
        """Add a module to modules.
        """
        module = modules.setdefault(mname, {})
        if only:  # Empty list means no ONLY clause
            for oname in only:
                module[oname] = True

    def sort_module_info(self, modules, module_name, imports=None):
        """Return USE statements based on modules.
        Save any names which must be imported in imports to be used with
        interface blocks.
        """
        arg_f_use = []
        for mname in sorted(modules):
            only = modules[mname]
            if mname == module_name:
                if imports is not None:
                    imports.update(only)
            else:
                if only:
                    snames = sorted(only.keys())
                    arg_f_use.append('use %s, only : %s' % (
                        mname, ', '.join(snames)))
                else:
                    arg_f_use.append('use %s' % mname)
        return arg_f_use

    def dump_generic_interfaces(self):
        """Generate code for generic interfaces into self.generic_interface
        """
        # Look for generic interfaces
        # splicer to extend generic
        self._push_splicer('generic')
        iface = self.generic_interface
        for key in sorted(self.f_function_generic.keys()):
            generics = self.f_function_generic[key]
            if len(generics) > 1:
                self._push_splicer(key)
                iface.append('')
                iface.append('interface ' + key)
                iface.append(1)
                for node in generics:
                    if node.cpp_if:
                        iface.append('#' + node.cpp_if)
                    iface.append('module procedure ' + node.fmtdict.F_name_impl)
                    if node.cpp_if:
                        iface.append('#endif')
                iface.append(-1)
                iface.append('end interface ' + key)
                self._pop_splicer(key)
        self._pop_splicer('generic')

    def add_abstract_interface(self, node, arg):
        """Record an abstract interface.

        Function pointers are converted to abstract interfaces.
        The interface is named after the function and the argument.
        """
        ast = node.ast
        fmt = util.Scope(node.fmtdict)
        fmt.argname = arg.name
        name = wformat(
            node.options.F_abstract_interface_subprogram_template, fmt)
        entry = self.f_abstract_interface.get(name)
        if entry is None:
            self.f_abstract_interface[name] = (node, fmt, arg)
        return name

    def dump_abstract_interfaces(self):
        """Generate code for abstract interfaces
        """
        self._push_splicer('abstract')
        if len(self.f_abstract_interface) > 0:
            iface = self.abstract_interface
            iface.append('')
            iface.append('abstract interface')
            iface.append(1)

            for key in sorted(self.f_abstract_interface.keys()):
                node, fmt, arg = self.f_abstract_interface[key]
                ast = node.ast
                subprogram = arg.get_subprogram()
                iface.append('')
                arg_f_names = []
                arg_c_decl = []
                modules = {}   # indexed as [module][variable]
                for i, param in enumerate(arg.params):
                    name = param.name
                    if name is None:
                        fmt.index = str(i)
                        name = wformat(
                            node.options.F_abstract_interface_argument_template, fmt)
                    arg_f_names.append(name)
                    arg_c_decl.append(param.bind_c(name=name))

                    arg_typedef, c_statements = typemap.lookup_c_statements(param)
                    self.update_f_module(modules,
                                         arg_typedef.f_c_module or arg_typedef.f_module)

                if subprogram == 'function':
                    arg_c_decl.append(ast.bind_c(name=key, params=None))
                arguments = ',\t '.join(arg_f_names)
                iface.append('{} {}({}) bind(C)'.format(
                    subprogram, key, arguments))
                iface.append(1)
                arg_f_use = self.sort_module_info(modules, None)
                iface.extend(arg_f_use)
                iface.append('implicit none')
                iface.extend(arg_c_decl)
                iface.append(-1)
                iface.append('end {} {}'.format(subprogram, key))
            iface.append(-1)
            iface.append('')
            iface.append('end interface')
        self._pop_splicer('abstract')

    def wrap_function_interface(self, cls, node):
        """
        Write Fortran interface for C function
        cls  - class node or None for functions
        node - function/method node

        Wrapping involves both a C interface and a Fortran wrapper.
        For some generic functions there may be single C method with
        multiple Fortran wrappers.
        """
        options = node.options
        fmt_func = node.fmtdict
        fmt = util.Scope(fmt_func)

        ast = node.ast
        result_type = ast.typename
        is_ctor = ast.fattrs.get('_constructor', False)
        is_dtor = ast.fattrs.get('_destructor', False)
        is_pure = ast.fattrs.get('pure', False)
        is_static = False
        is_allocatable = ast.fattrs.get('allocatable', False)
        func_is_const = ast.func_const
        subprogram = ast.get_subprogram()

        if node._generated == 'arg_to_buffer':
            generated_suffix = '_buf'
        else:
            generated_suffix = ''

        if is_dtor or node.return_this:
            result_type = 'void'
            subprogram = 'subroutine'
        elif fmt_func.C_custom_return_type:
            result_type = fmt_func.C_custom_return_type
            subprogram = 'function'

        result_typedef = typemap.Typedef.lookup(result_type)

        arg_c_names = []  # argument names for functions
        arg_c_decl = []   # declaraion of argument names
        modules = {}   # indexed as [module][variable]
        imports = {}   # indexed as [name]

        # find subprogram type
        # compute first to get order of arguments correct.
        # Add
        if subprogram == 'subroutine':
            fmt.F_C_subprogram = 'subroutine'
        else:
            fmt.F_C_subprogram = 'function'
            fmt.F_C_result_clause = '\fresult(%s)' % fmt.F_result

        if cls:
            is_static = 'static' in ast.storage
            if is_ctor or is_static:
                pass
            else:
                # Add 'this' argument
                arg_c_names.append(fmt.C_this)
                arg_c_decl.append(
                    'type(C_PTR), value, intent(IN) :: ' + fmt.C_this)
                self.set_f_module(modules, 'iso_c_binding', 'C_PTR')

        args_all_in = True   # assume all arguments are intent(in)
        for arg in ast.params:
            # default argument's intent
            # XXX look at const, ptr
            arg_typedef, c_statements = typemap.lookup_c_statements(arg)
            fmt.c_var = arg.name
            attrs = arg.attrs
            self.update_f_module(modules,
                                 arg_typedef.f_c_module or arg_typedef.f_module)

            intent = attrs.get('intent', 'inout')
            if intent != 'in':
                args_all_in = False

            # argument names
            if arg_typedef.f_c_args:
                for argname in arg_typedef.f_c_args:
                    arg_c_names.append(argname)
            else:
                arg_c_names.append(arg.name)

            # argument declarations
            if attrs.get('_is_result', False) and is_allocatable:
                arg_c_decl.append(
                    'type(C_PTR), intent(OUT) :: {}'.format(
                        arg.name))
            elif arg.is_function_pointer():
                absiface = self.add_abstract_interface(node, arg)
                arg_c_decl.append(
                    'procedure({}) :: {}'.format(
                        absiface, arg.name))
                imports[absiface] = True
            elif arg_typedef.f_c_argdecl:
                for argdecl in arg_typedef.f_c_argdecl:
                    append_format(arg_c_decl, argdecl, fmt)
            else:
                arg_c_decl.append(arg.bind_c())

            if attrs.get('_is_result', False):
                c_stmts = 'result' + generated_suffix
            else:
                c_stmts = 'intent_' + intent + generated_suffix

            c_intent_blk = c_statements.get(c_stmts, {})

            # Add implied buffer arguments to prototype
            for buf_arg in c_intent_blk.get('buf_args', []):
                if buf_arg not in attrs:
                    raise RuntimeError("{} is missing from {} for {}"
                                       .format(buf_arg,
                                               str(c_intent_blk['buf_args']),
                                               node.declgen))
                buf_arg_name = attrs[buf_arg]
                if buf_arg == 'size':
                    arg_c_names.append(buf_arg_name)
                    arg_c_decl.append(
                        'integer(C_LONG), value, intent(IN) :: %s' % buf_arg_name)
                    self.set_f_module(modules, 'iso_c_binding', 'C_LONG')
                elif buf_arg == 'len_trim':
                    arg_c_names.append(buf_arg_name)
                    arg_c_decl.append(
                        'integer(C_INT), value, intent(IN) :: %s' % buf_arg_name)
                    self.set_f_module(modules, 'iso_c_binding', 'C_INT')
                elif buf_arg == 'len':
                    arg_c_names.append(buf_arg_name)
                    arg_c_decl.append(
                        'integer(C_INT), value, intent(IN) :: %s' % buf_arg_name)
                    self.set_f_module(modules, 'iso_c_binding', 'C_INT')
                elif buf_arg == 'lenout':
                    # result of allocatable std::string or std::vector
                    arg_c_names.append(buf_arg_name)
                    arg_c_decl.append(
                        'integer(C_SIZE_T), intent(OUT) :: %s' % buf_arg_name)
                    self.set_f_module(modules, 'iso_c_binding', 'C_SIZE_T')
                else:
                    raise RuntimeError("wrap_function_interface: unhandled case {}"
                                       .format(buf_arg))

        if (subprogram == 'function' and
                (is_pure or (func_is_const and args_all_in))):
            fmt.F_C_pure_clause = 'pure '

        fmt.F_C_arguments = options.get(
            'F_C_arguments', ',\t '.join(arg_c_names))

        if fmt.F_C_subprogram == 'function':
            if result_typedef.base == 'string':
                arg_c_decl.append('type(C_PTR) %s' % fmt.F_result)
                self.set_f_module(modules, 'iso_c_binding', 'C_PTR')
            else:
                # XXX - make sure ptr is set to avoid VALUE
                rvast = declast.create_this_arg(fmt.F_result, result_type, False)
                arg_c_decl.append(rvast.bind_c())
                self.update_f_module(modules,
                                     result_typedef.f_c_module or
                                     result_typedef.f_module)

        arg_f_use = self.sort_module_info(modules, fmt_func.F_module_name, imports)

        c_interface = self.c_interface
        c_interface.append('')

        if node.cpp_if:
            c_interface.append('#' + node.cpp_if)
        c_interface.append(
            wformat('\r{F_C_pure_clause}{F_C_subprogram} {F_C_name}'
                    '(\t{F_C_arguments}){F_C_result_clause}'
                    '\fbind(C, name="{C_name}")', fmt))
        c_interface.append(1)
        c_interface.extend(arg_f_use)
        if imports:
            c_interface.append('import :: ' + ', '.join(sorted(imports.keys())))
        c_interface.append('implicit none')
        c_interface.extend(arg_c_decl)
        c_interface.append(-1)
        c_interface.append(wformat('end {F_C_subprogram} {F_C_name}', fmt))
        if node.cpp_if:
            c_interface.append('#endif')

    def attr_allocatable(self, allocatable, node, arg, pre_call):
        """Add the allocatable attribute to the pre_call block.

        Valid values of allocatable:
           mold=name
        """
        fmtargs = node._fmtargs

        p = re.compile('mold\s*=\s*(\w+)')
        m = p.match(allocatable)
        if m is not None:
            moldvar = m.group(1)
            if moldvar not in fmtargs:
                raise RuntimeError("Mold argument {} does not exist: {}"
                                   .format(moldvar, allocatable))
            for moldarg in node.ast.params:
                if moldarg.name == moldvar:
                    break
            if 'dimension' not in moldarg.attrs:
                raise RuntimeError("Mold argument {} must have dimension attribute"
                                   .format(moldvar))
            fmt = fmtargs[arg.name]['fmtf']
            if True:
                rank = len(moldarg.attrs['dimension'].split(','))
                bounds = []
                for i in range(1, rank+1):
                    bounds.append('lbound({var},{dim}):ubound({var},{dim})'.
                                  format(var=moldvar, dim=i))
                fmt.mold = ','.join(bounds)  
                append_format(pre_call, 'allocate({f_var}({mold}))', fmt)
            else:
                # f2008 supports the mold option which makes this easier
                fmt.mold = m.group(0)
                append_format(pre_call, 'allocate({f_var}, {mold})', fmt)

    def attr_implied(self, node, arg, fmt):
        """Add the implied attribute to the pre_call block.
        """
        init = arg.attrs.get('implied', None)
        blk = {}
        if init:
            fmt.pre_call_intent = ftn_implied(init, node, arg)
            blk['pre_call'] = [
                '{f_var} = {pre_call_intent}'
            ]
        return blk

    def wrap_function_impl(self, cls, node):
        """
        Wrap implementation of Fortran function
        """
        options = node.options
        fmt_func = node.fmtdict

        # Assume that the C function can be called directly.
        # If the wrapper does any work, then set need_wraper to True
        need_wrapper = options['F_force_wrapper']
        if node._overloaded:
            # need wrapper for generic interface
            need_wrapper = True

        # Look for C routine to wrap
        # Usually the same node unless it is a generic function
        C_node = node
        generated = []
        if C_node._generated:
            generated.append(C_node._generated)
        while C_node._PTR_F_C_index is not None:
            C_node = self.newlibrary.function_index[C_node._PTR_F_C_index]
            if C_node._generated:
                generated.append(C_node._generated)
#  #This is no longer true with the result as an argument
#        if len(node.params) != len(C_node.params):
#            raise RuntimeError("Argument mismatch between Fortran and C functions")

        fmt_func.F_C_call = C_node.fmtdict.F_C_name
        fmtargs = C_node._fmtargs

        # Fortran return type
        ast = node.ast
        result_type = ast.typename
        is_ctor = ast.fattrs.get('_constructor', False)
        is_dtor = ast.fattrs.get('_destructor', False)
        is_pure = ast.fattrs.get('pure', False)
        is_static = False
        is_allocatable = ast.fattrs.get('allocatable', False)
        subprogram = ast.get_subprogram()
        c_subprogram = C_node.ast.get_subprogram()

        if C_node._generated == 'arg_to_buffer':
            generated_suffix = '_buf'
        else:
            generated_suffix = ''

        if is_dtor or node.return_this:
            result_type = 'void'
            subprogram = 'subroutine'
            c_subprogram = 'subroutine'
        elif fmt_func.C_custom_return_type:
            # User has changed the return type of the C function
            # TODO: probably needs to be more clever about
            # setting pointer or reference fields too.
            # Maybe parse result_type instead of copy.
            result_type = fmt_func.C_custom_return_type
            subprogram = 'function'
            c_subprogram = 'function'
            ast = copy.deepcopy(node.ast)
            ast.typename = result_type

        result_typedef = typemap.Typedef.lookup(result_type)
        if not result_typedef:
            raise RuntimeError("Unknown type {} in {}",
                               result_type, fmt_func.function_name)

        result_generated_suffix = ''
        if is_pure:
            result_generated_suffix = '_pure'

        # this catches stuff like a bool to logical conversion which
        # requires the wrapper
        if result_typedef.f_statements.get('result' + result_generated_suffix, {}) \
                                      .get('need_wrapper', False):
            need_wrapper = True

        arg_c_call = []      # arguments to C function

        arg_f_names = []
        arg_f_decl = []
        modules = {}   # indexed as [module][variable]

        if subprogram == 'function':
            fmt_func.F_result_clause = '\fresult(%s)' % fmt_func.F_result
        fmt_func.F_subprogram = subprogram

        if cls:
            need_wrapper = True
            is_static = 'static' in ast.storage
            if is_ctor or is_static:
                pass
            else:
                # Add 'this' argument
                # could use {f_to_c} but I'd rather not hide the shadow class
                arg_c_call.append(wformat('{F_this}%{F_derived_member}', fmt_func))
                arg_f_names.append(fmt_func.F_this)
                arg_f_decl.append(wformat(
                        'class({F_derived_name}) :: {F_this}',
                        fmt_func))

        # Fortran and C arguments may have different types (fortran generic)
        #
        # f_var - argument to Fortran function (wrapper function)
        # c_var - argument to C function (wrapped function)
        #
        # May be one more argument to C function than Fortran function
        # (the result)
        #
        pre_call = []
        post_call = []
        f_args = ast.params
        f_index = -1       # index into f_args
        for c_arg in C_node.ast.params:
            arg_name = c_arg.name
            fmt_arg0 = fmtargs.setdefault(arg_name, {})
            fmt_arg  = fmt_arg0.setdefault('fmtf', util.Scope(fmt_func))
            fmt_arg.f_var = arg_name
            fmt_arg.c_var = arg_name

            is_f_arg = True   # assume C and Fortran arguments match
            c_attrs = c_arg.attrs
            allocatable = c_attrs.get('allocatable', False)
            implied = c_attrs.get('implied', False)
            intent = c_attrs['intent']
            allocatable_result = False  # XXX - kludgeish

            # string C functions may have their results copied
            # into an argument passed in, F_string_result_as_arg.
            # Or the wrapper may provide an argument in the Fortran API
            # to hold the result.
            if c_attrs.get('_is_result', False):
                # XXX - _is_result implies a string result for now
                # This argument is the C function result
                if is_allocatable:
                    allocatable_result = True
                c_stmts = 'result' + generated_suffix
                f_stmts = 'result' + generated_suffix
                if not fmt_func.F_string_result_as_arg:
                    # It is not in the Fortran API
                    is_f_arg = False
                    fmt_arg.c_var = fmt_func.F_result
                    fmt_arg.f_var = fmt_func.F_result
                    need_wrapper = True
            else:
                c_stmts = 'intent_' + intent + generated_suffix
                f_stmts = 'intent_' + intent

            if is_f_arg:
                # An argument to the C and Fortran function
                f_index += 1
                f_arg = f_args[f_index]
                if f_arg.is_function_pointer():
                    absiface = self.add_abstract_interface(node, f_arg)
                    arg_f_decl.append(
                        'procedure({}) :: {}'.format(
                            absiface, f_arg.name))
                    arg_f_names.append(fmt_arg.f_var)
                    # function pointers are pass thru without any change
                    arg_c_call.append(f_arg.name)
                    continue
                elif implied:
                    # An implied argument is not passed into Fortran
                    # it is computed then passed to C++
                    arg_f_decl.append(f_arg.gen_arg_as_fortran(local=True))
                else:
                    arg_f_decl.append(f_arg.gen_arg_as_fortran())
                    arg_f_names.append(fmt_arg.f_var)
            else:
                # Pass result as an argument to the C++ function.
                f_arg = c_arg
                if allocatable_result:
                    # character allocatable function
                    fmt_arg.f_cptr = 'SHP_' + arg_name
                    append_format(arg_f_decl, 'type(C_PTR) :: {f_cptr}',
                                  fmt_arg)
                    self.set_f_module(modules, 'iso_c_binding', 'C_PTR')

            arg_type = f_arg.typename
            arg_typedef = typemap.Typedef.lookup(arg_type)
            base_typedef = arg_typedef
            if 'template' in c_attrs:
                # If a template, use its type
                cxx_T = c_attrs['template']
                arg_typedef = typemap.Typedef.lookup(cxx_T)

            self.update_f_module(modules, arg_typedef.f_module)

            if implied:
                f_intent_blk = self.attr_implied(node, f_arg, fmt_arg)
            else:
                f_statements = arg_typedef.f_statements
                f_intent_blk = f_statements.get(f_stmts, {})

            # Now C function arguments
            # May have different types, like generic
            # or different attributes, like adding +len to string args
            arg_typedef = typemap.Typedef.lookup(c_arg.typename)
            arg_typedef, c_statements = typemap.lookup_c_statements(c_arg)
            c_intent_blk = c_statements.get(c_stmts, {})

            # Create a local variable for C if necessary
            have_c_local_var = f_intent_blk.get('c_local_var', False)
            if have_c_local_var:
                fmt_arg.c_var = 'SH_' + fmt_arg.f_var
                arg_f_decl.append('{} {}'.format(
                    arg_typedef.f_c_type or arg_typedef.f_type, fmt_arg.c_var))

            # Attributes   None=skip, True=use default, else use value
            if allocatable_result:
                arg_c_call.append(fmt_arg.f_cptr)
            elif arg_typedef.f_args:
                # TODO - Not sure if this is still needed.
                need_wrapper = True
                append_format(arg_c_call, arg_typedef.f_args, fmt_arg)
            elif arg_typedef.f_to_c:
                need_wrapper = True
                append_format(arg_c_call, arg_typedef.f_to_c, fmt_arg)
            elif f_arg and c_arg.typename != f_arg.typename:
                need_wrapper = True
                append_format(arg_c_call, arg_typedef.f_cast, fmt_arg)
                self.update_f_module(modules, arg_typedef.f_module)
            else:
                arg_c_call.append(fmt_arg.c_var)

            # Add any buffer arguments
            for buf_arg in c_intent_blk.get('buf_args', []):
                need_wrapper = True
                buf_arg_name = c_attrs[buf_arg]
                if buf_arg == 'size':
                    append_format(arg_c_call, 'size({f_var}, kind=C_LONG)', fmt_arg)
                    self.set_f_module(modules, 'iso_c_binding', 'C_LONG')
                elif buf_arg == 'len_trim':
                    append_format(arg_c_call, 'len_trim({f_var}, kind=C_INT)', fmt_arg)
                    self.set_f_module(modules, 'iso_c_binding', 'C_INT')
                elif buf_arg == 'len':
                    append_format(arg_c_call, 'len({f_var}, kind=C_INT)', fmt_arg)
                    self.set_f_module(modules, 'iso_c_binding', 'C_INT')
                elif buf_arg == 'lenout':
                    fmt_arg.f_var_len = c_attrs['lenout']
                    append_format(arg_f_decl, 'integer(C_SIZE_T) :: {f_var_len}',
                                  fmt_arg)
                    append_format(arg_c_call, '{f_var_len}', fmt_arg)
                    self.set_f_module(modules, 'iso_c_binding', 'C_SIZE_T')
                else:
                    raise RuntimeError("wrap_function_impl: unhandled case {}"
                                       .format(buf_arg))

            # Add code for intent of argument
            cmd_list = f_intent_blk.get('declare', [])
            if cmd_list:
                need_wrapper = True
                for cmd in cmd_list:
                    append_format(arg_f_decl, cmd, fmt_arg)

            cmd_list = f_intent_blk.get('pre_call', [])
            if cmd_list:
                need_wrapper = True
                for cmd in cmd_list:
                    append_format(pre_call, cmd, fmt_arg)

            cmd_list = f_intent_blk.get('post_call', [])
            if cmd_list:
                need_wrapper = True
                for cmd in cmd_list:
                    append_format(post_call, cmd, fmt_arg)

            # Find any helper routines needed
            if 'f_helper' in f_intent_blk:
                for helper in f_intent_blk['f_helper'].split():
                    self.f_helper[helper] = True

            if allocatable:
                attr_allocatable(allocatable, C_node, f_arg, pre_call)

        # use tabs to insert continuations
        fmt_func.F_arg_c_call = ',\t '.join(arg_c_call)
        fmt_func.F_arguments = options.get('F_arguments', ',\t '.join(arg_f_names))

        # declare function return value after arguments
        # since arguments may be used to compute return value
        # (for example, string lengths)
        if subprogram == 'function':
            # if func_is_const:
            #     fmt_func.F_pure_clause = 'pure '
            if result_typedef.base == 'string':
                if is_allocatable:
                    append_format(arg_f_decl,
                                  'character(len=:,kind=C_CHAR), allocatable :: {F_result}',
                                  fmt_func)
                else:
                    # special case returning a string
                    rvlen = ast.fattrs.get('len', None)
                    if rvlen is None:
                        rvlen = wformat(
                            'strlen_ptr(\t{F_C_call}(\t{F_arg_c_call}))',
                            fmt_func)
                    else:
                        rvlen = str(rvlen)  # convert integers
                    fmt_func.c_var_len = wformat(rvlen, fmt_func)
                    line1 = wformat(
                        'character(kind=C_CHAR,\t len={c_var_len})\t :: {F_result}',
                        fmt_func)
                    arg_f_decl.append(line1)
                self.set_f_module(modules, 'iso_c_binding', 'C_CHAR')
            else:
                arg_f_decl.append(ast.gen_arg_as_fortran(name=fmt_func.F_result))
            self.update_f_module(modules, result_typedef.f_module)

        if not node._CXX_return_templated:
            # if return type is templated in C++,
            # then do not set up generic since only the
            # return type may be different (ex. getValue<T>())
            if cls and not is_ctor:
                self.f_type_generic.setdefault(
                    fmt_func.F_name_generic, []).append(node)
            else:
                self.f_function_generic.setdefault(
                    fmt_func.class_prefix + fmt_func.F_name_generic, []).append(node)
        if cls:
            # Add procedure to derived type
            if is_static:
                self.type_bound_part.append('procedure, nopass :: %s => %s' % (
                    fmt_func.F_name_function, fmt_func.F_name_impl))
            elif not is_ctor:
                self.type_bound_part.append('procedure :: %s => %s' % (
                    fmt_func.F_name_function, fmt_func.F_name_impl))

        # body of function
        # XXX sname = fmt_func.F_name_impl
        sname = fmt_func.F_name_function
        splicer_code = self.splicer_stack[-1].get(sname, None)
        if fmt_func.inlocal('F_code'):
            need_wrapper = True
            F_code = [wformat(fmt_func.F_code, fmt_func)]
        elif splicer_code:
            need_wrapper = True
            F_code = splicer_code
        else:
            F_code = []
            if is_ctor:
                fmt_func.F_call_code = wformat(
                    '{F_result}%{F_derived_member} = '
                    '{F_C_call}({F_arg_c_call})', fmt_func)
                F_code.append(fmt_func.F_call_code)
            elif c_subprogram == 'function':
                f_statements = result_typedef.f_statements
                intent_blk = f_statements.get('result' + result_generated_suffix,{})
                cmd_list = intent_blk.get('call', [
                        '{F_result} = {F_C_call}({F_arg_c_call})'])
#                for cmd in cmd_list:  # only allow a single statment for now
#                    append_format(pre_call, cmd, fmt_arg)
                fmt_func.F_call_code = wformat(cmd_list[0], fmt_func)
                F_code.append(fmt_func.F_call_code)

                # Find any helper routines needed
                if 'f_helper' in intent_blk:
                    for helper in intent_blk['f_helper'].split():
                        self.f_helper[helper] = True
            else:
                fmt_func.F_call_code = wformat('call {F_C_call}({F_arg_c_call})', fmt_func)
                F_code.append(fmt_func.F_call_code)

#            if result_typedef.f_post_call:
#                need_wrapper = True
#                # adjust return value or cleanup
#                append_format(F_code, result_typedef.f_post_call, fmt_func)
            if is_dtor:
                F_code.append(wformat(
                    '{F_this}%{F_derived_member} = C_NULL_PTR', fmt_func))
                self.set_f_module(modules, 'iso_c_binding', 'C_NULL_PTR')

        arg_f_use = self.sort_module_info(modules, fmt_func.F_module_name)

        if need_wrapper:
            impl = self.impl
            impl.append('')
            if node.cpp_if:
                impl.append('#' + node.cpp_if)
            if options.debug:
                impl.append('! %s' % node.declgen)
                if generated:
                    impl.append('! %s' % ' - '.join(generated))
                impl.append('! function_index=%d' % node._function_index)
                if options.doxygen and node.doxygen:
                    self.write_doxygen(impl, node.doxygen)
            impl.append(
                wformat('\r{F_subprogram} {F_name_impl}(\t'
                        '{F_arguments}){F_result_clause}',
                        fmt_func))
            impl.append(1)
            impl.extend(arg_f_use)
            impl.extend(arg_f_decl)
            impl.extend(pre_call)
            self._create_splicer(sname, impl, F_code)
            impl.extend(post_call)
            impl.append(-1)
            impl.append(wformat('end {F_subprogram} {F_name_impl}', fmt_func))
            if node.cpp_if:
                impl.append('#endif')
        else:
            fmt_func.F_C_name = fmt_func.F_name_impl

    def write_module(self, library, cls):
        """ Write Fortran wrapper module.
        """
        node = cls or library
        options = node.options
        fmt_node = node.fmtdict
        fname = fmt_node.F_impl_filename
        module_name = fmt_node.F_module_name

        output = []

        if options.doxygen:
            self.write_doxygen_file(output, fname, library, cls)
        self._create_splicer('file_top', output)

        output.append('module %s' % module_name)
        output.append(1)

        # Write use statments (classes use iso_c_binding C_PTR)
        arg_f_use = self.sort_module_info(self.module_use, module_name)
        output.extend(arg_f_use)
        self.module_use = {}

        if options.F_module_per_class:
            output.extend(self.use_stmts)
        else:
            self._create_splicer('module_use', output)
        output.append('implicit none')
        output.append('')
        if cls is None:
            self._create_splicer('module_top', output)

        output.extend(self.enum_impl)

        # XXX output.append('! splicer push class')
        output.extend(self.f_type_decl)
        # XXX  output.append('! splicer pop class')

        # Interfaces for operator overloads
        if self.operator_map:
            ops = sorted(self.operator_map)
            for op in ops:
                output.append('')
                output.append('interface operator (%s)' % op)
                output.append(1)
                for opfcn in self.operator_map[op]:
                    output.append('module procedure %s' % opfcn)
                output.append(-1)
                output.append('end interface')

        self.dump_abstract_interfaces()
        self.dump_generic_interfaces()

        output.extend(self.abstract_interface)
        output.extend(self.c_interface)
        output.extend(self.generic_interface)

        # Insert any helper functions needed
        # (They are duplicated in each module)
        helper_source = []
        if self.f_helper:
            helperdict = whelpers.find_all_helpers('f', self.f_helper)
            helpers = sorted(self.f_helper)
            private_names = []
            interface_lines = []
            for helper in helpers:
                helper_info = helperdict[helper]
                private_names.extend(helper_info.get('private', []))
                lines = helper_info.get('interface', None)
                if lines:
                    interface_lines.append(lines)
                lines = helper_info.get('source', None)
                if lines:
                    helper_source.append(lines)
            if private_names:
                output.append('')
                output.append('private ' + ', '.join(private_names))
            output.extend(interface_lines)

        output.append(-1)
        output.append('')
        output.append('contains')
        output.append(1)

        output.extend(self.impl)

        output.extend(self.operator_impl)

        output.extend(helper_source)

        output.append(-1)
        output.append('')
        output.append('end module %s' % module_name)

        self.config.ffiles.append(
            os.path.join(self.config.c_fortran_dir, fname))
        self.write_output_file(fname, self.config.c_fortran_dir, output)

    def write_c_helper(self):
        """ Write C helper functions that will be used by the wrappers.
        """
        pass

class ToImplied(todict.PrintNode):
    """Convert implied expression to Python wrapper code.

    expression has already been checked for errors by generate.check_implied.
    Convert functions:
      size  -  PyArray_SIZE
    """
    def __init__(self, expr, func, arg):
        super(ToImplied, self).__init__()
        self.expr = expr
        self.func = func
        self.arg = arg

    def visit_Identifier(self, node):
        # Look for functions
        if node.args == None:
            return node.name
        elif node.name == 'size':
            # size(arg)
            # This expected to be assigned to a C_INT or C_LONG
            # add KIND argument to the size intrinsic
            argname = node.args[0].name
            arg_typedef = typemap.Typedef.lookup(self.arg.typename)
            return 'size({},kind={})'.format(argname, arg_typedef.f_kind)
        else:
            return self.param_list(node)

def ftn_implied(expr, func, arg):
    """Convert string to Fortran code.
    """
    node = declast.ExprParser(expr).expression()
    visitor = ToImplied(expr, func, arg)
    return visitor.visit(node)


def attr_allocatable(allocatable, node, arg, pre_call):
    """Add the allocatable attribute to the pre_call block.

    Valid values of allocatable:
       mold=name
    """
    fmtargs = node._fmtargs

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
        fmt = fmtargs[arg.name]['fmtf']
        if node.options.F_standard >= 2008:
            # f2008 supports the mold option which makes this easier
            fmt.mold = m.group(0)
            append_format(pre_call, 'allocate({f_var}, {mold})', fmt)
        else:
            rank = len(moldarg.attrs['dimension'].split(','))
            bounds = []
            for i in range(1, rank+1):
                bounds.append('lbound({var},{dim}):ubound({var},{dim})'.
                              format(var=moldvar, dim=i))
            fmt.mold = ','.join(bounds)  
            append_format(pre_call, 'allocate({f_var}({mold}))', fmt)

