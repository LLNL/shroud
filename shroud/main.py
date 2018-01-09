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
generate language bindings
"""

#
# Annotate the YAML tree with additional internal fields
#  _decl            - generated declaration.
#                     Includes computed attributes
#  _function_index  - sequence number function,
#                     used in lieu of a pointer
#  _generated       - who generated this function
#  _PTR_F_C_index   - Used by fortran wrapper to find index of
#                     C function to call
#  _PTR_C_CPP_index - Used by C wrapper to find index of C++ function
#                     to call
#  _subprogram      - subroutine or function
#
#
from __future__ import print_function
from __future__ import absolute_import

import argparse
import copy
import json
import os
import sys
import yaml

from . import ast
from . import declast
from . import splicer
from . import typemap
from . import util
from . import wrapc
from . import wrapf
from . import wrapp
from . import wrapl
from . import whelpers

wformat = util.wformat


class Config(object):
    """A class to stash configuration values.
    """
    pass


class Schema(object):
    """
    Verify that the input dictionary has the correct fields.
    Create defaults for missing fields.


    check_schema
      check_classes
        check_class
          check_function
        check_class_depedencies
          check_function_dependencies
      check_functions
        check_function
    """
    def __init__(self, tree, config):
        self.tree = tree    # json tree
        self.config = config
        self.fmt_stack = []

    def push_options(self, node):
        """ Push a new set of options.
        Copy current options, then update with new options.
        Replace node[option] dictionary with Options instance.
        Return original options dictionary.
        """
        old = None
        new = util.Options(parent=self.options_stack[-1])
        if 'options' in node and \
                node['options'] is not None:
            if not isinstance(node['options'], dict):
                raise TypeError("options must be a dictionary")
            old = node['options']
            new.update(old)
        self.options_stack.append(new)
        node['options'] = new
        return new, old

    def pop_options(self):
        self.options_stack.pop()

    def check_options_only(self, node):
        """Process an options only entry in a list.

        Return True if node only has options.
        node is assumed to be a dictionary.
        Update current set of options from node['options'].
        """
        if len(node) != 1:
            return False
        options = node.get('options', None)
        if not options:
            return False
        if not isinstance(options, dict):
            raise TypeError("options must be a dictionary")

        # replace current options
        new = util.Options(parent=self.options_stack[-1])
        new.update(node['options'])
        self.options_stack[-1] = new
        return True

    def push_fmt(self, node):
        fmt = util.Options(self.fmt_stack[-1])
        self.fmt_stack.append(fmt)
        node['_fmt'] = fmt
        return fmt

    def pop_fmt(self):
        self.fmt_stack.pop()

    def check_schema(self):
        """ Check entire schema of input tree.
        Create format dictionaries.
        """
        node = self.tree

        def_types, def_types_alias = typemap.initialize()
        declast.add_typemap()

        # Write out as YAML if requested
        if self.config.yaml_types:
            with open(os.path.join(self.config.yaml_dir, self.config.yaml_types), 'w') as yaml_file:
                yaml.dump(def_types, yaml_file, default_flow_style=False)
            print("Wrote", self.config.yaml_types)

        if 'types' in node:
            types_dict = node['types']
            if not isinstance(types_dict, dict):
                raise TypeError("types must be a dictionary")
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

        # Add to node so they show up in the json debug file.
        node['_types'] = def_types
        node['_type_aliases'] = def_types_alias

        newlibrary = ast.LibraryNode(node)
        node['newlibrary'] = newlibrary

        # recreate old behavior for _fmt and options
        node['_fmt'] = node['newlibrary']._fmt
        self.fmt_stack.append(node['_fmt'])

        self.options_stack = [ newlibrary.options ]
        node['options'] = newlibrary.options

        patterns = node.setdefault('patterns', [])
        classes = node.setdefault('classes', [])
#        self.check_classes(classes)
#        self.check_functions(node, '', 'functions')
        # XXX - for json
        node['classes'] = newlibrary.classes
        node['functions'] = newlibrary.functions

    def check_classes(self, node):
        if not isinstance(node, list):
            raise TypeError("classes must be a list")
        for cls in node:
            if not isinstance(cls, dict):
                raise TypeError("classes[n] must be a dictionary")
            if 'name' not in cls:
                raise TypeError("class does not define name")
            declast.add_type(cls['name'])
        for cls in node:
            self.check_class(cls)

    def check_class(self, node):
        if 'name' not in node:
            raise RuntimeError('Expected name for class')
        name = node['name']

        # default cpp_header to blank
        if 'cpp_header' not in node:
            node['cpp_header'] = ''
        if node['cpp_header'] is None:
            # YAML turns blank strings into None
            node['cpp_header'] = ''

        options, old = self.push_options(node)
        fmt_class = self.push_fmt(node)
        self.option_to_fmt(fmt_class, old)
        fmt_class.cpp_class = name
        fmt_class.class_lower = name.lower()
        fmt_class.class_upper = name.upper()
        util.eval_template(node, 'class_prefix')

        # Only one file per class for C.
        util.eval_template(node, 'C_header_filename', '_class')
        util.eval_template(node, 'C_impl_filename', '_class')

        if options.F_module_per_class:
            util.eval_template(node, 'F_module_name', '_class')
            util.eval_template(node, 'F_impl_filename', '_class')

        self.check_functions(node, name, 'methods')
        self.pop_fmt()
        self.pop_options()


class GenFunctions(object):
    """
    Generate types from class.
    Generate functions based on overload/template/generic/attributes
    Computes fmt.function_suffix.
    """

    def __init__(self, tree, config):
        self.tree = tree    # json tree
        self.config = config

    def gen_library(self):
        """Entry routine to generate functions for a library.
        """
        tree = self.tree
        newlibrary = self.tree['newlibrary']
#        tree = newlibrary

        # Order of creating.
        # Each is given a _function_index when created.
        self.function_index = []
        newlibrary['function_index'] = self.function_index

        for cls in newlibrary.classes:
            cls.functions = self.define_function_suffix(cls.functions)
        newlibrary['functions'] = self.define_function_suffix(newlibrary.functions)
        tree['functions'] = newlibrary.functions # XXX - for json

# No longer need this, but keep code for now in case some other dependency checking is needed
#        for cls in tree['classes']:
#            self.check_class_dependencies(cls)

    def append_function_index(self, node):
        """append to function_index, set index into node.
        """
        ilist = self.function_index
        node['_function_index'] = len(ilist)
#        node['_fmt'].function_index = str(len(ilist)) # debugging
        ilist.append(node)

    def define_function_suffix(self, functions):
        """
        Return a new list with generated function inserted.
        """

        # Look for overloaded functions
        cpp_overload = {}
        for function in functions:
            if function.function_suffix is not None:
                function._fmt.function_suffix = function.function_suffix
            self.append_function_index(function)
            cpp_overload. \
                setdefault(function._ast.name, []). \
                append(function['_function_index'])

        # keep track of which function are overloaded in C++.
        for key, value in cpp_overload.items():
            if len(value) > 1:
                for index in value:
                    self.function_index[index]['_cpp_overload'] = value

        # Create additional functions needed for wrapping
        ordered_functions = []
        for method in functions:
            if method._has_default_arg:
                self.has_default_args(method, ordered_functions)
            ordered_functions.append(method)
            if method.cpp_template:
                method._overloaded = True
                self.template_function(method, ordered_functions)

        # Look for overloaded functions
        overloaded_functions = {}
        for function in ordered_functions:
            # if not function.options.wrap_c:
            #     continue
            if function.cpp_template:
                continue
            overloaded_functions.setdefault(
                function._ast.name, []).append(function)

        # look for function overload and compute function_suffix
        for mname, overloads in overloaded_functions.items():
            if len(overloads) > 1:
                for i, function in enumerate(overloads):
                    function['_overloaded'] = True
                    if not function._fmt.inlocal('function_suffix'):
                        function._fmt.function_suffix = '_{}'.format(i)

        # Create additional C bufferify functions.
        ordered3 = []
        for method in ordered_functions:
            ordered3.append(method)
            self.arg_to_buffer(method, ordered3)

        # Create multiple generic Fortran wrappers to call a
        # single C functions
        ordered4 = []
        for method in ordered3:
            ordered4.append(method)
            if not method.options.wrap_fortran:
                continue
            if method.fortran_generic:
                method._overloaded = True
                self.generic_function(method, ordered4)

        self.gen_functions_decl(ordered4)

        return ordered4

    def template_function(self, node, ordered_functions):
        """ Create overloaded functions for each templated argument.
        """
        if len(node.cpp_template) != 1:
            # In the future it may be useful to have multiple templates
            # That the would start creating more permutations
            raise NotImplementedError("Only one cpp_templated type for now")
        for typename, types in node.cpp_template.items():
            for type in types:
                new = util.copy_function_node(node)
                ordered_functions.append(new)
                self.append_function_index(new)

                new['_generated'] = 'cpp_template'
                fmt = new._fmt
                fmt.function_suffix = fmt.function_suffix + '_' + type
                new.cpp_template = {}
                options = new.options
                options.wrap_c = True
                options.wrap_fortran = True
                options.wrap_python = False
                options.wrap_lua = False
                # Convert typename to type
                fmt.CPP_template = '<{}>'.format(type)
                if new._ast.typename == typename:
                    new._ast.typename = type
                    new._CPP_return_templated = True
                for arg in new._ast.params:
                    if arg.typename == typename:
                        arg.typename = type

        # Do not process templated node, instead process
        # generated functions above.
        options = node.options
        options.wrap_c = False
        options.wrap_fortran = False
        options.wrap_python = False
        options.wrap_lua = False

    def generic_function(self, node, ordered_functions):
        """ Create overloaded functions for each generic method.
        """
        if len(node.fortran_generic) != 1:
            # In the future it may be useful to have multiple generic arguments
            # That the would start creating more permutations
            raise NotImplemented("Only one generic arg for now")
        for argname, types in node.fortran_generic.items():
            for type in types:
                new = util.copy_function_node(node)
                ordered_functions.append(new)
                self.append_function_index(new)

                new['_generated'] = 'fortran_generic'
                new['_PTR_F_C_index'] = node['_function_index']
                fmt = new._fmt
                # XXX append to existing suffix
                fmt.function_suffix = fmt.function_suffix + '_' + type
                new.fortran_generic = {}
                options = new.options
                options.wrap_c = False
                options.wrap_fortran = True
                options.wrap_python = False
                options.wrap_lua = False
                # Convert typename to type
                for arg in new._ast.params:
                    if arg.name == argname:
                        # Convert any typedef to native type with f_type
                        argtype = arg.typename
                        typedef = typemap.Typedef.lookup(argtype)
                        typedef = typemap.Typedef.lookup(typedef.f_type)
                        if not typedef.f_cast:
                            raise RuntimeError(
                                "unable to cast type {} in fortran_generic"
                                .format(argtype))
                        arg.typename = type

        # Do not process templated node, instead process
        # generated functions above.
        options = node.options
#        options.wrap_c = False
        options.wrap_fortran = False
#        options.wrap_python = False

    def has_default_args(self, node, ordered_functions):
        """
        For each function which has a default argument, generate
        a version for each possible call.
          void func(int i = 0, int j = 0)
        generates
          void func()
          void func(int i)
          void func(int i, int j)
        """
        default_funcs = []

        default_arg_suffix = node.default_arg_suffix
        ndefault = 0

        min_args = 0
        for i, arg in enumerate(node._ast.params):
            if arg.init is None:
                min_args += 1
                continue
            new = util.copy_function_node(node)
            self.append_function_index(new)
            new['_generated'] = 'has_default_arg'
            del new._ast.params[i:]  # remove trailing arguments
            new['_has_default_arg'] = False
            options = new.options
            options.wrap_c = True
            options.wrap_fortran = True
            options.wrap_python = False
            options.wrap_lua = False
            fmt = new._fmt
            try:
                fmt.function_suffix = default_arg_suffix[ndefault]
            except IndexError:
                # XXX fmt.function_suffix =
                # XXX  fmt.function_suffix + '_nargs%d' % (i + 1)
                pass
            default_funcs.append(new['_function_index'])
            ordered_functions.append(new)
            ndefault += 1

        # keep track of generated default value functions
        node['_default_funcs'] = default_funcs
        node['_nargs'] = (min_args, len(node._ast.params))
        # The last name calls with all arguments (the original decl)
        try:
            node._fmt.function_suffix = default_arg_suffix[ndefault]
        except IndexError:
            pass

    def arg_to_buffer(self, node, ordered_functions):
        """Look for function which have implied arguments.
        This includes functions with string or vector arguments.
        If found then create a new C function that
        will convert argument into a buffer and length.
        """
        options = node.options
        fmt = node._fmt

        # If a C++ function returns a std::string instance,
        # the default wrapper will not compile since the wrapper
        # will be declared as char. It will also want to return the
        # c_str of a stack variable. Warn and turn off the wrapper.
        ast = node._ast
        result_type = ast.typename
        result_typedef = typemap.Typedef.lookup(result_type)
        # wrapped classes have not been added yet.
        # Only care about string here.
        attrs = ast.attrs
        result_is_ptr = ast.is_indirect()
        if result_typedef and result_typedef.base in ['string', 'vector'] and \
                result_type != 'char' and \
                not result_is_ptr:
            options.wrap_c = False
#            options.wrap_fortran = False
            self.config.log.write("Skipping {}, unable to create C wrapper "
                                  "for function returning {} instance"
                                  " (must return a pointer or reference).\n"
                                  .format(result_typedef.cpp_type,
                                          ast.name))

        if options.wrap_fortran is False:
            return
        if options.F_string_len_trim is False:  # XXX what about vector
            return

        # Is result or any argument a string?
        has_implied_arg = False
        for arg in ast.params:
            argtype = arg.typename
            typedef = typemap.Typedef.lookup(argtype)
            if typedef.base == 'string':
                is_ptr = arg.is_indirect()
                if is_ptr:
                    has_implied_arg = True
                else:
                    arg.typename = 'char_scalar'
            elif typedef.base == 'vector':
                has_implied_arg = True

        has_string_result = False
        result_as_arg = ''  # only applies to string functions
        is_pure = ast.fattrs.get('pure', False)
        if result_typedef.base == 'vector':
            raise NotImplemented("vector result")
        elif result_typedef.base == 'string':
            if result_type == 'char' and not result_is_ptr:
                # char functions cannot be wrapped directly in intel 15.
                ast.typename = 'char_scalar'
            has_string_result = True
            result_as_arg = fmt.F_string_result_as_arg
            result_name = result_as_arg or fmt.C_string_result_as_arg

        if not (has_string_result or has_implied_arg):
            return

        # XXX       options = node['options']
        # XXX       options.wrap_fortran = False
        # Preserve wrap_c.
        # This keep a version which accepts char * arguments.

        # Create a new C function and change arguments
        # to add len_trim attribute
        C_new = util.copy_function_node(node)
        ordered_functions.append(C_new)
        self.append_function_index(C_new)

        C_new._generated = 'arg_to_buffer'
        C_new._error_pattern_suffix = '_as_buffer'
        fmt = C_new._fmt
        fmt.function_suffix = fmt.function_suffix + options.C_bufferify_suffix

        options = C_new.options
        options.wrap_c = True
        options.wrap_fortran = False
        options.wrap_python = False
        options.wrap_lua = False
        C_new._PTR_C_CPP_index = node._function_index

        newargs = []
        for arg in C_new._ast.params:
            attrs = arg.attrs
            argtype = arg.typename
            arg_typedef = typemap.Typedef.lookup(argtype)
            if arg_typedef.base == 'vector':
                # Do not wrap the orignal C function with vector argument.
                # Meaningless to call without the size argument.
                # TODO: add an option where char** length is determined by looking
                #       for trailing NULL pointer.  { "foo", "bar", NULL };
                node.options.wrap_c = False
                node.options.wrap_python = False  # NotImplemented
                node.options.wrap_lua = False     # NotImplemented
            arg_typedef, c_statements = typemap.lookup_c_statements(arg)

            # set names for implied buffer arguments
            stmts = 'intent_' + attrs['intent'] + '_buf'
            intent_blk = c_statements.get(stmts, {})
            for buf_arg in intent_blk.get('buf_args', []):
                if buf_arg in attrs:
                    # do not override user specified variable name
                    continue
                if buf_arg == 'size':
                    attrs['size'] = options.C_var_size_template.format(
                        c_var=arg.name)
                elif buf_arg == 'len_trim':
                    attrs['len_trim'] = options.C_var_trim_template.format(
                        c_var=arg.name)
                elif buf_arg == 'len':
                    attrs['len'] = options.C_var_len_template.format(
                        c_var=arg.name)

                ## base typedef

        # Copy over some buffer specific fields to their generic name.
        C_new.C_post_call = C_new.C_post_call_buf

        if has_string_result:
            # Add additional argument to hold result
            ast = C_new._ast
            result_as_string = ast.result_as_arg(result_name)
            attrs = result_as_string.attrs
            attrs['len'] = options.C_var_len_template.format(c_var=result_name)
            attrs['intent'] = 'out'
            attrs['_is_result'] = True
            # convert to subroutine
            C_new['_subprogram'] = 'subroutine'

        if is_pure:
            # pure functions which return a string have result_pure defined.
            pass
        elif result_as_arg:
            # Create Fortran function without bufferify function_suffix but
            # with len attributes on string arguments.
            F_new = util.copy_function_node(C_new)
            ordered_functions.append(F_new)
            self.append_function_index(F_new)

            # Fortran function should wrap the new C function
            F_new['_PTR_F_C_index'] = C_new['_function_index']
            options = F_new.options
            options.wrap_c = False
            options.wrap_fortran = True
            options.wrap_python = False
            options.wrap_lua = False
            # Do not add '_bufferify'
            F_new._fmt.function_suffix = node._fmt.function_suffix

            # Do not wrap original function (does not have result argument)
            node.options.wrap_fortran = False
        else:
            # Fortran function may call C subroutine if string result
            node['_PTR_F_C_index'] = C_new['_function_index']

    def XXXcheck_class_dependencies(self, node):
        """
        Check used_types and find which header and module files
        to use for this class
        """
        # keep track of types which are used by methods arguments
        used_types = {}
        for method in node['methods']:
            self.check_function_dependencies(method, used_types)

        modules = {}
        for typ in used_types.values():
            if typ.f_module:
                for mname, only in typ.f_module.items():
                    module = modules.setdefault(mname, {})
                    if only:  # Empty list means no ONLY clause
                        for oname in only:
                            module[oname] = True

        # Always add C_PTR, needed for class F_derived_member
        modules.setdefault('iso_c_binding', {})['C_PTR'] = True

        F_modules = []  # array of tuples ( name, (only1, only2) )
        for mname in sorted(modules):
            F_modules.append((mname, sorted(modules[mname])))
        node['F_module_dependencies'] = F_modules

    def XXXcheck_function_dependencies(self, node, used_types):
        """Record which types are used by a function.
        """
        if node.cpp_template:
            # The templated type will raise an error.
            # XXX - Maybe dummy it out
            # XXX - process templated types
            return
        ast = node._ast
        rv_type = ast.typename
        typedef = typemap.Typedef.lookup(rv_type)
        if typedef is None:
            raise RuntimeError(
                "Unknown type {} for function decl: {}"
                .format(rv_type, node['decl']))
        result_typedef = typemap.Typedef.lookup(rv_type)
        # XXX - make sure it exists
        used_types[rv_type] = result_typedef
        for arg in ast.params:
            argtype = arg.typename
            typedef = typemap.Typedef.lookup(argtype)
            if typedef is None:
                raise RuntimeError("%s not defined" % argtype)
            if typedef.base == 'wrapped':
                used_types[argtype] = typedef

    def gen_functions_decl(self, functions):
        """ Generate _decl for generated all functions.
        """
        for node in functions:
            node._decl = node._ast.gen_decl()


class VerifyAttrs(object):
    """
    Check attributes and set some defaults.
    Generate types for classes.
    """
    def __init__(self, tree, config):
        self.tree = tree    # json tree
        self.config = config

    def verify_attrs(self):
        tree = self.tree
        newlibrary = self.tree['newlibrary']

        for cls in newlibrary.classes:
            typemap.create_class_typedef(cls)

        for cls in newlibrary.classes:
            for func in cls.functions:
                self.check_arg_attrs(func)

        for func in newlibrary.functions:
            self.check_arg_attrs(func)

    def check_arg_attrs(self, node):
        """Regularize attributes
        intent: lower case, no parens, must be in, out, or inout
        value: if pointer, default to False (pass-by-reference;
               else True (pass-by-value).
        """
        options = node.options
        if not options.wrap_fortran and not options.wrap_c:
            return

        # cache subprogram type
        ast = node._ast
        result_type = ast.typename
        result_is_ptr = ast.is_pointer()
        #  'void'=subroutine   'void *'=function
        if result_type == 'void' and not result_is_ptr:
            node._subprogram = 'subroutine'
        else:
            node._subprogram = 'function'

        found_default = False
        for arg in ast.params:
            argname = arg.name
            argtype = arg.typename
            typedef = typemap.Typedef.lookup(argtype)
            if typedef is None:
                # if the type does not exist, make sure it is defined by cpp_template
                #- decl: void Function7(ArgType arg)
                #  cpp_template:
                #    ArgType:
                #    - int
                #    - double
                if argtype not in node.cpp_template:
                    raise RuntimeError("No such type %s: %s" % (
                            argtype, arg.gen_decl()))

            is_ptr = arg.is_indirect()
            attrs = arg.attrs

            # intent
            intent = attrs.get('intent', None)
            if intent is None:
                if not is_ptr:
                    attrs['intent'] = 'in'
                elif arg.const:
                    attrs['intent'] = 'in'
                elif typedef.base == 'string':
                    attrs['intent'] = 'inout'
                elif typedef.base == 'vector':
                    attrs['intent'] = 'inout'
                else:
                    # void *
                    attrs['intent'] = 'in'  # XXX must coordinate with VALUE
            else:
                intent = intent.lower()
                if intent in ['in', 'out', 'inout']:
                    attrs['intent'] = intent
                else:
                    raise RuntimeError(
                        "Bad value for intent: " + attrs['intent'])

            # value
            value = attrs.get('value', None)
            if value is None:
                if is_ptr:
                    if (typedef.f_c_type or typedef.f_type) == 'type(C_PTR)':
                        # This causes Fortran to dereference the C_PTR
                        # Otherwise a void * argument becomes void **
                        attrs['value'] = True
                    else:
                        attrs['value'] = False
                else:
                    attrs['value'] = True

            # dimension
            dimension = attrs.get('dimension', None)
            if dimension:
                if attrs.get('value', False):
                    raise RuntimeError("argument must not have value=True")
                if not is_ptr:
                    raise RuntimeError("dimension attribute can only be "
                                       "used on pointer and references")
                if dimension is True:
                    # No value was provided, provide default
                    attrs['dimension'] = '(*)'
                else:
                    # Put parens around dimension
                    attrs['dimension'] = '(' + attrs['dimension'] + ')'
            elif typedef and typedef.base == 'vector':
                # default to 1-d assumed shape 
                attrs['dimension'] = '(:)'

            if arg.init is not None:
                found_default = True
                node['_has_default_arg'] = True
            elif found_default is True:
                raise RuntimeError("Expected default value for %s" % argname)

            # compute argument names for some attributes
            # XXX make sure they don't conflict with other names
            len_name = attrs.get('len', False)
            if len_name is True:
                attrs['len'] = options.C_var_len_template.format(c_var=argname)
            len_name = attrs.get('len_trim', False)
            if len_name is True:
                attrs['len_trim'] = options.C_var_trim_template.format(c_var=argname)
            size_name = attrs.get('size', False)
            if size_name is True:
                attrs['size'] = options.C_var_size_template.format(c_var=argname)

            # Check template attribute
            temp = attrs.get('template', None)
            if typedef and typedef.base == 'vector':
                if not temp:
                    raise RuntimeError("std::vector must have template argument: %s" % (
                            arg.gen_decl()))
                typedef = typemap.Typedef.lookup(temp)
                if typedef is None:
                    raise RuntimeError("No such type %s for template: %s" % (
                            temp, arg.gen_decl()))
            elif temp is not None:
                raise RuntimeError("Type '%s' may not supply template argument: %s" % (
                        argtype, arg.gen_decl()))


class Namify(object):
    """Compute names of functions in library.
    Need to compute F_name and F_C_name since they interact.
    Compute all C names first, then Fortran.
    A Fortran function may call a generated C function via
    _PTR_F_C_index
    Also compute number which may be controlled by options.

    C_name - Name of C function
    F_C_name - Fortran function for C interface
    F_name_impl - Name of Fortran function implementation
    """
    def __init__(self, tree, config):
        self.tree = tree    # json tree
        self.config = config

    def name_library(self):
        self.name_language(self.name_function_c)
        self.name_language(self.name_function_fortran)

    def name_language(self, handler):
        tree = self.tree['newlibrary']
        for cls in tree['classes']:
            for func in cls.functions:
                handler(cls, func)

            options = cls.options
            fmt_class = cls._fmt
            if 'F_this' in options:
                fmt_class.F_this = options.F_this

        for func in tree['functions']:
            handler(None, func)

    def name_function_c(self, cls, node):
        options = node.options
        if not options.wrap_c:
            return
        fmt_func = node._fmt

        node.eval_template('C_name')
        node.eval_template('F_C_name')
        fmt_func.F_C_name = fmt_func.F_C_name.lower()

        if 'C_this' in options:
            fmt_func.C_this = options.C_this

    def name_function_fortran(self, cls, node):
        """ Must process C functions to generate their names.
        """
        options = node.options
        if not options.wrap_fortran:
            return
        fmt_func = node._fmt

        node.eval_template('F_name_impl')
        node.eval_template('F_name_function')
        node.eval_template('F_name_generic')

        if 'F_this' in options:
            fmt_func.F_this = options.F_this
        if 'F_result' in options:
            fmt_func.F_result = options.F_result


class TypeOut(util.WrapperMixin):
    """A class to write out type information.
    It subclasses util.WrapperMixin in order to access 
    write routines.
    """
    def __init__(self, tree, config):
        self.tree = tree    # json tree
        self.newlibrary = tree['newlibrary']
        self.config = config
        self.log = config.log
        self.comment = '#'

    def write_types(self):
        """Write out types into a file.
        This file can be read by Shroud to share types.
        """
        newlibrary = self.newlibrary
        newlibrary.eval_template('YAML_type_filename')
        fname = newlibrary._fmt.YAML_type_filename
        output = [
            '# Types generated by Shroud for class {}'.format(
                self.newlibrary['library']),
            'types:',
        ]

        write_file = False
        for cls in newlibrary.classes:
            name = cls.name
            output.append('')
            output.append('  {}:'.format(name))
            self.tree['_types'][name].__export_yaml__(2, output)
            write_file = True

            # yaml.dump does not make a nice output
            # line = yaml.dump(self.tree['types'][cls['name']],
            #                  default_flow_style=False)

        # debug prints
        # name = 'bool'
        # output.append('  {}:'.format(name))
        # self.tree['types'][name].__as_yaml__(2, output)

        if write_file:
            self.write_output_file(fname, self.config.yaml_dir, output)


def main():
    from . import __version__

    appname = 'shroud'

    parser = argparse.ArgumentParser(
        prog=appname,
        description="""Create Fortran or Python wrapper for a C++ library.
""")
    parser.add_argument('--version', action='version', version=__version__)
    parser.add_argument('--outdir', default='',
                        help='Directory for output files.'
                        'Defaults to current directory.')
    parser.add_argument('--outdir-c-fortran', default='',
                        dest='outdir_c_fortran',
                        help='Directory for C/Fortran wrapper output files, '
                        'overrides --outdir.')
    parser.add_argument('--outdir-python', default='',
                        dest='outdir_python',
                        help='Directory for Python wrapper output files, '
                        'overrides --outdir.')
    parser.add_argument('--outdir-lua', default='',
                        dest='outdir_lua',
                        help='Directory for Lua wrapper output files, '
                        'overrides --outdir.')
    parser.add_argument('--outdir-yaml', default='',
                        dest='outdir_yaml',
                        help='Directory for yaml output files, '
                        'overrides --outdir.')
    parser.add_argument('--logdir', default='',
                        help='Directory for log files.'
                        'Defaults to current directory.')
    parser.add_argument('--cfiles', default='',
                        help='Output file with list of C and C++ files '
                        'created.')
    parser.add_argument('--ffiles', default='',
                        help='Output file with list of Fortran created')
    parser.add_argument('--path', default=[], action='append',
                        help='Colon delimited paths to search for '
                        'splicer files, may be supplied multiple '
                        'times to append to path.')
    parser.add_argument('--cmake', default='',
                        help='Create a file with CMake macro')
    parser.add_argument('--yaml-types', default='',
                        help='Write a YAML file with default types')
    parser.add_argument('filename', nargs='*',
                        help='Input file to process.')

    args = parser.parse_args()
    main_with_args(args)
#    sys.stderr.write("Some useful message")  # example error message
    sys.exit(0)  # set status for errors


def main_with_args(args):
    """Main after args have been parsed.
    Useful for testing.
    """

    if args.cmake:
        # Create C make file
        try:
            fp = open(args.cmake, 'w')
            fp.write(whelpers.cmake)
            fp.close()
            raise SystemExit
        except IOError as e:
            print(str(e))
            raise SystemExit(1)

    # check command line options
    if len(args.filename) == 0:
        raise SystemExit("Must give at least one input file")
    if args.outdir and not os.path.isdir(args.outdir):
        raise SystemExit("outdir {} does not exist"
                         .format(args.outdir))
    if args.outdir_c_fortran and not os.path.isdir(args.outdir_c_fortran):
        raise SystemExit("outdir-fortran {} does not exist"
                         .format(args.outdir_c_fortran))
    if args.outdir_python and not os.path.isdir(args.outdir_python):
        raise SystemExit("outdir-python {} does not exist"
                         .format(args.outdir_python))
    if args.outdir_lua and not os.path.isdir(args.outdir_lua):
        raise SystemExit("outdir-lua {} does not exist"
                         .format(args.outdir_lua))
    if args.outdir_yaml and not os.path.isdir(args.outdir_yaml):
        raise SystemExit("outdir-yaml {} does not exist"
                         .format(args.outdir_yaml))
    if args.logdir and not os.path.isdir(args.logdir):
        raise SystemExit("logdir {} does not exist"
                         .format(args.logdir))

    # append all paths together
    if args.path:
        search_path = []
        for pth in args.path:
            search_path.extend(pth.split(':'))
    else:
        search_path = ['.']

    basename = os.path.splitext(os.path.basename(args.filename[0]))[0]
    logpath = os.path.join(args.logdir, basename + '.log')
    log = open(logpath, 'w')

    # pass around a configuration object
    config = Config()
    config.c_fortran_dir = args.outdir_c_fortran or args.outdir
    config.python_dir = args.outdir_python or args.outdir
    config.lua_dir = args.outdir_lua or args.outdir
    config.yaml_dir = args.outdir_yaml or args.outdir
    config.yaml_types = args.yaml_types
    config.log = log
    config.cfiles = []  # list of C/C++ files created
    config.ffiles = []  # list of Fortran files created

    # accumulated input
    all = dict(
        library='default_library',
        cpp_header='',
        namespace='',
        language='c++',
        )
    splicers = dict(c={}, f={}, py={}, lua={})

    for filename in args.filename:
        root, ext = os.path.splitext(filename)
        if ext == '.yaml':
#            print("Read %s" % os.path.basename(filename))
            log.write("Read yaml %s\n" % os.path.basename(filename))
            fp = open(filename, 'r')
            d = yaml.load(fp.read())
            fp.close()
            if d is not None:
                all.update(d)
#            util.update(all, d)  # recursive update
        elif ext == '.json':
            raise NotImplemented("Can not deal with json input for now")
        else:
            # process splicer file on command line, search path is not used
            splicer.get_splicer_based_on_suffix(filename, splicers)

#    print(all)

    Schema(all, config).check_schema()
    VerifyAttrs(all, config).verify_attrs()
    GenFunctions(all, config).gen_library()
    Namify(all, config).name_library()

    if 'splicer' in all:
        # read splicer files defined in input YAML file
        for suffix, names in all['splicer'].items():
            # suffix = 'c', 'f', 'py', 'lua'
            subsplicer = splicers.setdefault(suffix, {})
            for name in names:
                for pth in search_path:
                    fullname = os.path.join(pth, name)
#                    log.write("Try splicer %s\n" % fullname)
                    if os.path.isfile(fullname):
                        break
                else:
                    fullname = None
                if not fullname:
                    raise RuntimeError("File not found: %s" % name)
                log.write("Read splicer %s\n" % name)
                splicer.get_splicers(fullname, subsplicer)

    # Add any explicit splicers in the YAML file.
    if 'splicer_code' in all:
        splicers.update(all['splicer_code'])

    # Write out generated types
    TypeOut(all, config).write_types()

    try:
        options = all['options']
        if options.wrap_c:
            wrapc.Wrapc(all, config, splicers['c']).wrap_library()

        if options.wrap_fortran:
            wrapf.Wrapf(all, config, splicers['f']).wrap_library()

        if options.wrap_python:
            wrapp.Wrapp(all, config, splicers['py']).wrap_library()

        if options.wrap_lua:
            wrapl.Wrapl(all, config, splicers['lua']).wrap_library()
    finally:
        # Write a debug dump even if there was an exception.
        # when dumping json, remove function_index to avoid duplication
#        del all['function_index']

        jsonpath = os.path.join(args.logdir, basename + '.json')
        fp = open(jsonpath, 'w')

        # Test top level _fmt and options
        all['_fmt'] = all['newlibrary']._fmt
        all['options'] = all['newlibrary'].options

        json.dump(all, fp, cls=util.ExpandedEncoder, sort_keys=True, indent=4)
        fp.close()

    # Write list of output files.  May be useful for build systems
    if args.cfiles:
        with open(args.cfiles, 'w') as fp:
            if config.cfiles:
                fp.write(' '.join(config.cfiles))
            fp.write('\n')
    if args.ffiles:
        with open(args.ffiles, 'w') as fp:
            if config.ffiles:
                fp.write(' '.join(config.ffiles))
            fp.write('\n')

    log.close()

# This helps when running with a pipe, like CMake's execute_process
# It doesn't fix the error, but it does report a better error message
# http://www.thecodingforums.com/threads/help-with-a-piping-error.749747/
    sys.stdout.flush()


if __name__ == '__main__':
    main()
