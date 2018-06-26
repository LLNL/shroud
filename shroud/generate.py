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
Generate additional functions required to create wrappers.
"""
from __future__ import print_function
from __future__ import absolute_import

from . import ast
from . import declast
from . import todict
from . import typemap
from . import util
from . import whelpers

wformat = util.wformat

class VerifyAttrs(object):
    """
    Check attributes and set some defaults.
    Generate types for classes.
    """
    def __init__(self, newlibrary, config):
        self.newlibrary = newlibrary
        self.config = config

    def verify_attrs(self):
        newlibrary = self.newlibrary

        for cls in newlibrary.classes:
            if not cls.as_struct:
                for var in cls.variables:
                    self.check_var_attrs(cls, var)
            for func in cls.functions:
                self.check_fcn_attrs(func)

        for func in newlibrary.functions:
            self.check_fcn_attrs(func)

    def check_var_attrs(self, cls, node):
        for attr in node.ast.attrs:
            if attr[0] == '_': # internal attribute
                continue
            if attr not in [
                    'name',
                    'readonly',
            ]:
                raise RuntimeError(
                    "Illegal attribute '{}' for variable '{}' at line {}"
                    .format(attr, node.ast.name, node.linenumber))

    def check_fcn_attrs(self, node):
        options = node.options
        if not options.wrap_fortran and not options.wrap_c:
            return

        ast = node.ast
        node._has_found_default = False

        for attr in node.ast.attrs:
            if attr[0] == '_': # internal attribute
                continue
            if attr not in [
                    'allocatable',  # return a Fortran ALLOCATABLE
                    'deref',        # How to dereference pointer
                    'dimension',
                    'free_pattern',
                    'len',
                    'name',
                    'owner',
                    'pure',
            ]:
                raise RuntimeError(
                    "Illegal attribute '{}' for function '{}' define at line {}"
                    .format(attr, node.ast.name, node.linenumber))
        self.check_shared_attrs(node.ast)

        for arg in ast.params:
            self.check_arg_attrs(node, arg)

    def check_shared_attrs(self, node):
        """Check attributes which may be assigned to function or argument:
        deref, free_pattern, owner
        """
        attrs = node.attrs

        deref = attrs.get('deref', None)
        if deref is not None:
            if deref not in ['allocatable', 'pointer', 'raw', 'scalar']:
                raise RuntimeError("Illegal value '{}' for deref attribute. "
                                   "Must be 'allocatable', 'pointer', 'raw', "
                                   "or 'scalar'.".format(deref))
# XXX deref only on pointer, vector

        owner = attrs.get('owner', None)
        if owner is not None:
            if owner not in ['caller', 'library']:
                raise RuntimeError("Illegal value '{}' for owner attribute. "
                                   "Must be 'caller' or 'library'.".format(deref))

        free_pattern = attrs.get('free_pattern', None)
        if free_pattern is not None:
            if free_pattern not in self.newlibrary.patterns:
                raise RuntimeError("Illegal value '{}' for free_pattern attribute. "
                                   "Must be defined in patterns section."
                                   .format(free_pattern))

    def check_arg_attrs(self, node, arg):
        """Regularize attributes
        intent: lower case, no parens, must be in, out, or inout
        value: if pointer, default to False (pass-by-reference;
               else True (pass-by-value).
        """
        argname = arg.name

        for attr in arg.attrs:
            if attr[0] == '_': # internal attribute
                continue
            if attr not in [
                    'allocatable',
                    'capsule',
                    'deref',
                    'dimension',
                    'hidden',  # omitted in Fortran API, returned from C++
                    'implied', # omitted in Fortran API, value passed to C++
                    'intent',
                    'len', 'len_trim', 'size',
                    'template',
                    'value',
            ]:
                raise RuntimeError(
                    "Illegal attribute '{}' for argument '{}' defined at line {}"
                    .format(attr, argname, node.linenumber))

        argtype = arg.typename
        arg_typemap = typemap.lookup_type(argtype)
        if arg_typemap is None:
            # if the type does not exist, make sure it is defined by cxx_template
            #- decl: void Function7(ArgType arg)
            #  cxx_template:
            #    ArgType:
            #    - int
            #    - double
            if argtype not in node.cxx_template:
                raise RuntimeError("check_arg_attrs: No such type %s: %s" % (
                    argtype, node.decl))

        self.check_shared_attrs(arg)

        is_ptr = arg.is_indirect()
        attrs = arg.attrs

        allocatable = attrs.get('allocatable', False)
        if allocatable:
            if not is_ptr:
                raise RuntimeError("Allocatable may only be used with pointer variables")

        # intent
        intent = attrs.get('intent', None)
        if intent is None:
            if node is None:
                # do not default intent for function pointers
                pass
            elif not is_ptr:
                attrs['intent'] = 'in'
            elif arg.const:
                attrs['intent'] = 'in'
            elif arg_typemap.base == 'string':
                attrs['intent'] = 'inout'
            elif arg_typemap.base == 'vector':
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
                if (arg_typemap.f_c_type or arg_typemap.f_type) == 'type(C_PTR)':
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
                raise RuntimeError("argument must not have value=True"
                                   "because it has the dimension attribute.")
            if not is_ptr:
                raise RuntimeError("dimension attribute can only be "
                                   "used on pointer and references")
            if dimension is True:
                # No value was provided, provide default
                if 'allocatable' in attrs:
                    attrs['dimension'] = ':'
                else:
                    attrs['dimension'] = '*'
        elif arg_typemap and arg_typemap.base == 'vector':
            # default to 1-d assumed shape
            attrs['dimension'] = ':'

        if node:
            if arg.init is not None:
                node._has_default_arg = True
            elif node._has_found_default is True:
                raise RuntimeError("Expected default value for %s" % argname)

            if 'implied' in attrs:
                check_implied(attrs['implied'], node)

        # compute argument names for some attributes
        # XXX make sure they don't conflict with other names
        capsule_name = attrs.get('capsule', False)
        if capsule_name is True:
            attrs['capsule'] = options.C_var_capsule_template.format(c_var=argname)
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
        if arg_typemap and arg_typemap.base == 'vector':
            if not temp:
                raise RuntimeError("std::vector must have template argument: %s" % (
                    arg.gen_decl()))
            arg_typemap = typemap.lookup_type(temp)
            if arg_typemap is None:
                raise RuntimeError("check_arg_attr: No such type %s for template: %s" % (
                    temp, arg.gen_decl()))
        elif temp is not None:
            raise RuntimeError("Type '%s' may not supply template argument: %s" % (
                argtype, arg.gen_decl()))

        if arg.is_function_pointer():
            for arg1 in arg.params:
                self.check_arg_attrs(None, arg1)


class GenFunctions(object):
    """
    Generate Typemap from class.
    Generate functions based on overload/template/generic/attributes
    Computes fmt.function_suffix.
    """

    def __init__(self, newlibrary, config):
        self.newlibrary = newlibrary
        self.config = config

    def gen_library(self):
        """Entry routine to generate functions for a library.
        """
        newlibrary = self.newlibrary
        whelpers.add_external_helpers(newlibrary.fmtdict)
        whelpers.add_capsule_helper(newlibrary.fmtdict)

        self.function_index = newlibrary.function_index

        self.instantiate_classes(newlibrary)

        for cls in newlibrary.classes:
#            added = self.default_ctor_and_dtor(cls)
            if not cls.as_struct:
                for var in cls.variables:
                    self.add_var_getter_setter(cls, var)
            cls.functions = self.define_function_suffix(cls.functions)
        newlibrary.functions = self.define_function_suffix(newlibrary.functions)

# No longer need this, but keep code for now in case some other dependency checking is needed
#        for cls in newlibrary.classes:
#            self.check_class_dependencies(cls)

    def append_function_index(self, node):
        """append to function_index, set index into node.
        """
        ilist = self.function_index
        node._function_index = len(ilist)
#        node.fmtdict.function_index = str(len(ilist)) # debugging
        ilist.append(node)

    def XXX_default_ctor_and_dtor(self, cls):
        """Wrap default constructor and destructor.

        Needed when the ctor or dtor is not explicily in the input.

        XXX - do not wrap by default because they may be private.
        """
        found_ctor = False
        found_dtor = False
        for node in cls.functions:
            fattrs = node.ast.attrs
            found_ctor = found_ctor or fattrs.get('_constructor', False)
            found_dtor = found_dtor or fattrs.get('_destructor', False)

        if found_ctor and found_dtor:
            return cls.functions

        added = cls.functions[:]

        if not found_ctor:
            added.append(ast.FunctionNode(
                '{}()'.format(cls.name), parent=cls))
        if not found_dtor:
            added.append(ast.FunctionNode(
                '~{}()'.format(cls.name), parent=cls))

        return added

    def add_var_getter_setter(self, cls, var):
        """Create getter/setter functions for class variables.
        This allows wrappers to access class members.

        Do not wrap for Python since descriptors are created for
        class member variables.
        """
        ast = var.ast
        arg_typemap = typemap.lookup_type(ast.typename)
        fieldname = ast.name  # attrs.get('name', ast.name)

        fmt = util.Scope(var.fmtdict)

        options = dict(
            wrap_lua=False,
            wrap_python=False,
        )

        # getter
        funcname = 'get' + fieldname.capitalize()
        argdecl = ast.gen_arg_as_c(name=funcname, continuation=True)
        decl = '{}()'.format(argdecl)
        field = wformat('{CXX_this}->{field_name}', fmt)
        if arg_typemap.cxx_to_c is None:
            val = field
        else:
            fmt.cxx_var = field
            val = wformat(arg_typemap.cxx_to_c, fmt)
        return_val = 'return ' + val + ';'

        format = dict(
            C_code='{C_pre_call}\n' + return_val,
        )

        cls.add_function(decl, format=format, options=options)

        # setter
        if ast.attrs.get('readonly', False):
            return
        funcname = 'set' + ast.name.capitalize()
        argdecl = ast.gen_arg_as_c(name='val', continuation=True)
        decl = 'void {}({})'.format(funcname, argdecl)
        field = wformat('{CXX_this}->{field_name}', fmt)
        if arg_typemap.c_to_cxx is None:
            val = 'val'
        else:
            fmt.c_var = 'val'
            val = wformat(arg_typemap.c_to_cxx, fmt)
        set_val = '{} = {};'.format(field, val)

        attrs = dict(
            val=dict(
                intent='in',
                value=True,  # XXX - what about pointer variables?
            )
        )

        format = dict(
            C_code='{C_pre_call}\n' + set_val + '\nreturn;',
        )

        cls.add_function(decl, attrs=attrs, format=format, options=options)

    def instantiate_classes(self, node):
        """Instantate any template_arguments.
        node - LibraryNode or ClassNode.

        Create a new list of classes replacing 
        any class with template_arguments with instantiated classes.
        """
        clslist = []
        for cls in node.classes:
            if cls.template_arguments:
                # Replace class with new class for each template instantiation.
                for i, args in enumerate(cls.template_arguments):
                    newcls = cls.clone()

                    # Update name of class.
                    #  cxx_class - vector_0      (Fortran and C names)
                    #  cxx_type  - vector<int>
                    cxx_class = "{}_{}".format(newcls.fmtdict.cxx_class, i)
                    cxx_type = "{}{}".format(newcls.fmtdict.cxx_class,
                                             args.instantiation)

                    newcls.fmtdict.update(dict(
                        cxx_type=cxx_type,
                        cxx_class=cxx_class,
                        class_lower=cxx_class.lower(),
                        class_upper=cxx_class.upper(),
                        class_scope=cxx_class + '::',
                        F_derived_name=cxx_class.lower(),
                    ))

                    newcls.typemap = typemap.create_class_typemap(newcls)

                    # class_lower class_prefix class_upper cxx_class
#                    newcls.functions = self.define_function_suffix(newcls.functions)
                    clslist.append(newcls)
            else:
#                cls.functions = self.define_function_suffix(cls.functions)
                clslist.append(cls)
        node.classes = clslist

    def define_function_suffix(self, functions):
        """
        Return a new list with generated function inserted.

        functions - list of functions
        """

        # Look for overloaded functions
        cxx_overload = {}
        for function in functions:
            self.append_function_index(function)
            cxx_overload. \
                setdefault(function.ast.name, []). \
                append(function._function_index)

        # keep track of which function are overloaded in C++.
        for key, value in cxx_overload.items():
            if len(value) > 1:
                for index in value:
                    self.function_index[index]._cxx_overload = value

        # Create additional functions needed for wrapping
        ordered_functions = []
        for method in functions:
            if method._has_default_arg:
                self.has_default_args(method, ordered_functions)
            ordered_functions.append(method)
            if method.cxx_template:
                method._overloaded = True
                self.template_function(method, ordered_functions)

        # Look for overloaded functions
        overloaded_functions = {}
        for function in ordered_functions:
            # if not function.options.wrap_c:
            #     continue
            if function.cxx_template:
                continue
            overloaded_functions.setdefault(
                function.ast.name, []).append(function)

        # look for function overload and compute function_suffix
        for mname, overloads in overloaded_functions.items():
            if len(overloads) > 1:
                for i, function in enumerate(overloads):
                    function._overloaded = True
                    if not function.fmtdict.inlocal('function_suffix'):
                        function.fmtdict.function_suffix = '_{}'.format(i)

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

        - decl: void Function7(ArgType arg)
          cxx_template:
            ArgType:
            - int
            - double
        """
        oldoptions = node.options

        nkeys = 0
        for typename, types in node.cxx_template.items():
            if typename == '__line__':
                continue
            nkeys += 1
            if nkeys > 1:
                # In the future it may be useful to have multiple templates
                # That the would start creating more permutations
                raise NotImplementedError("Only one cxx_templated type for now",
                                          node.cxx_template)
            for type in types:
                new = node.clone()
                ordered_functions.append(new)
                self.append_function_index(new)

                new._generated = 'cxx_template'
                fmt = new.fmtdict
                fmt.function_suffix = fmt.function_suffix + '_' + type
                new.cxx_template = {}
                options = new.options
                options.wrap_c = oldoptions.wrap_c
                options.wrap_fortran = oldoptions.wrap_fortran
                options.wrap_python = oldoptions.wrap_python
                options.wrap_lua = oldoptions.wrap_lua
                # Convert typename to type
                fmt.CXX_template = '<{}>'.format(type)
                if new.ast.typename == typename:
                    new.ast.typename = type
                    new._CXX_return_templated = True
                for arg in new.ast.params:
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

        - decl: void Function9(double arg)
          fortran_generic:
            arg:
            - float
            - double
        """
        nkeys = 0
        for argname, types in node.fortran_generic.items():
            if argname == '__line__':
                continue
            nkeys += 1
            if nkeys > 1:
                # In the future it may be useful to have multiple generic arguments
                # That the would start creating more permutations
                raise NotImplementedError("Only one generic arg for now")
            for type in types:
                new = node.clone()
                ordered_functions.append(new)
                self.append_function_index(new)

                new._generated = 'fortran_generic'
                new._PTR_F_C_index = node._function_index
                fmt = new.fmtdict
                # XXX append to existing suffix
                fmt.function_suffix = fmt.function_suffix + '_' + type
                new.fortran_generic = {}
                options = new.options
                options.wrap_c = False
                options.wrap_fortran = True
                options.wrap_python = False
                options.wrap_lua = False
                # Convert typename to type
                for arg in new.ast.params:
                    if arg.name == argname:
                        # Convert any arg_typemap to native type with f_type
                        argtype = arg.typename
                        arg_typemap = typemap.lookup_type(argtype)
                        if not arg_typemap.f_cast:
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
        for i, arg in enumerate(node.ast.params):
            if arg.init is None:
                min_args += 1
                continue
            new = node.clone()
            self.append_function_index(new)
            new._generated = 'has_default_arg'
            del new.ast.params[i:]  # remove trailing arguments
            new._has_default_arg = False
            options = new.options
            options.wrap_c = True
            options.wrap_fortran = True
            # Python and Lua both deal with default args in their own way
            options.wrap_python = False
            options.wrap_lua = False
            fmt = new.fmtdict
            try:
                fmt.function_suffix = default_arg_suffix[ndefault]
            except IndexError:
                # XXX fmt.function_suffix =
                # XXX  fmt.function_suffix + '_nargs%d' % (i + 1)
                pass
            default_funcs.append(new._function_index)
            ordered_functions.append(new)
            ndefault += 1

        # keep track of generated default value functions
        node._default_funcs = default_funcs
        node._nargs = (min_args, len(node.ast.params))
        # The last name calls with all arguments (the original decl)
        try:
            node.fmtdict.function_suffix = default_arg_suffix[ndefault]
        except IndexError:
            pass

    def move_arg_attributes(self, attrs, old_node, new_node):
        """After new_node has been created from old_node,
        the result is being converted into an argument.
        Move some attributes that are associated with the function
        to the new argument.

        If deref is not set, then default to allocatable.

        attrs - attributes of the new argument
        old_node - The FunctionNode of the original function.
        new_node - The FunctionNode of the new function with
        """
        c_attrs = new_node.ast.attrs
        f_attrs = old_node.ast.attrs
        if 'deref' not in f_attrs:
            f_attrs['deref'] = 'allocatable'
            attrs['deref'] = 'allocatable'
        for name in ['owner', 'free_pattern']:
            if name in c_attrs:
                attrs[name] = c_attrs[name]
                del c_attrs[name]

    def arg_to_buffer(self, node, ordered_functions):
        """Look for function which have implied arguments.
        This includes functions with string or vector arguments.
        If found then create a new C function that
        will add arguments buf_args (typically a buffer and length).

        String arguments add deref(allocatable) by default so that
        char * will create an allocatable string in Fortran.

        """
        options = node.options
        fmt = node.fmtdict

        # If a C++ function returns a std::string instance,
        # the default wrapper will not compile since the wrapper
        # will be declared as char. It will also want to return the
        # c_str of a stack variable. Warn and turn off the wrapper.
        ast = node.ast
        result_type = ast.typename
        result_typemap = typemap.lookup_type(result_type)
        # shadow classes have not been added yet.
        # Only care about string here.
        attrs = ast.attrs
        result_is_ptr = ast.is_indirect()
        if result_typemap and \
           result_typemap.base in ['string', 'vector'] and \
           result_type != 'char' and \
           not result_is_ptr:
            options.wrap_c = False
#            options.wrap_fortran = False
            self.config.log.write("Skipping {}, unable to create C wrapper "
                                  "for function returning {} instance"
                                  " (must return a pointer or reference).\n"
                                  .format(result_typemap.cxx_type,
                                          ast.name))

        if options.wrap_fortran is False:
            # The buffer function is intended to be called by Fortran.
            # No Fortran, no need for buffer function.
            return
        if options.F_string_len_trim is False:  # XXX what about vector?
            return

        # Is result or any argument a string or vector?
        # If so, additional arguments will be passed down so
        # create buffer version of function.
        has_implied_arg = False
        for arg in ast.params:
            argtype = arg.typename
            arg_typemap = typemap.lookup_type(argtype)
            if arg_typemap.base == 'string':
                is_ptr = arg.is_indirect()
                if is_ptr:
                    has_implied_arg = True
                else:
                    arg.typename = 'char_scalar'
            elif arg_typemap.base == 'vector':
                has_implied_arg = True
                # Create helpers for vector template.
                cxx_T = arg.attrs['template']
                tempate_typemap = typemap.lookup_type(cxx_T)
                whelpers.add_copy_array_helper(dict(
                    cxx_type=cxx_T,
                    f_kind=tempate_typemap.f_kind,
                    C_prefix=fmt.C_prefix,
                    C_array_type=fmt.C_array_type,
                    F_array_type=fmt.F_array_type,
                    stdlib=fmt.stdlib,
                ))

        has_string_result = False
        has_allocatable_result = False
        result_as_arg = ''  # only applies to string functions
        if result_typemap.base == 'vector':
            raise NotImplementedError("vector result")
        elif result_typemap.base == 'string':
            if result_type == 'char' and not result_is_ptr:
                # char functions cannot be wrapped directly in intel 15.
                ast.typename = 'char_scalar'
            has_string_result = True
            result_as_arg = fmt.F_string_result_as_arg
            result_name = result_as_arg or fmt.C_string_result_as_arg
        elif result_is_ptr and attrs.get('deref', '') == 'allocatable':
            has_allocatable_result = True

        if not (has_string_result or has_allocatable_result or has_implied_arg):
            return

        # XXX       options = node['options']
        # XXX       options.wrap_fortran = False
        # Preserve wrap_c.
        # This keep a version which accepts char * arguments.

        # Create a new C function and change arguments
        # and add attributes.
        C_new = node.clone()
        ordered_functions.append(C_new)
        self.append_function_index(C_new)

        C_new._generated = 'arg_to_buffer'
        C_new.generated_suffix = '_buf'  # used to lookup c_statements
        fmt = C_new.fmtdict
        fmt.function_suffix = fmt.function_suffix + fmt.C_bufferify_suffix

        options = C_new.options
        options.wrap_c = True
        options.wrap_fortran = False
        options.wrap_python = False
        options.wrap_lua = False
        C_new._PTR_C_CXX_index = node._function_index

        for arg in C_new.ast.params:
            attrs = arg.attrs
            argtype = arg.typename
            arg_typemap = typemap.lookup_type(argtype)
            if arg_typemap.base == 'vector':
                # Do not wrap the orignal C function with vector argument.
                # Meaningless to call without the size argument.
                # TODO: add an option where char** length is determined by looking
                #       for trailing NULL pointer.  { "foo", "bar", NULL };
                node.options.wrap_c = False
                node.options.wrap_python = False  # NotImplemented
                node.options.wrap_lua = False     # NotImplemented
            arg_typemap, c_statements = typemap.lookup_c_statements(arg)

            # set names for implied buffer arguments
            stmts = 'intent_' + attrs['intent'] + '_buf'
            if stmts in c_statements:
                arg.attrs['_generated_suffix'] = '_buf'

            intent_blk = c_statements.get(stmts, {})
            for buf_arg in intent_blk.get('buf_args', []):
                if buf_arg in attrs:
                    # do not override user specified variable name
                    continue
                if buf_arg == 'size':
                    attrs['size'] = options.C_var_size_template.format(
                        c_var=arg.name)
                elif buf_arg == 'capsule':
                    attrs['capsule'] = options.C_var_capsule_template.format(
                        c_var=arg.name)
                elif buf_arg == 'context':
                    attrs['context'] = options.C_var_context_template.format(
                        c_var=arg.name)
                elif buf_arg == 'len_trim':
                    attrs['len_trim'] = options.C_var_trim_template.format(
                        c_var=arg.name)
                elif buf_arg == 'len':
                    attrs['len'] = options.C_var_len_template.format(
                        c_var=arg.name)

                ## base typemap

        if has_string_result:
            # Add additional argument to hold result.
            # Default to deref(allocatable).
            # This will allocate a new character variable to hold the
            # results of the C++ function.
            ast = C_new.ast
            f_attrs = node.ast.attrs   # Fortran function attributes

            if 'len' in ast.attrs or result_as_arg:
                # +len implies copying into users buffer.
                result_as_string = ast.result_as_arg(result_name)
                attrs = result_as_string.attrs
                attrs['len'] = options.C_var_len_template.format(c_var=result_name)
                # Special case for wrapf.py
                f_attrs['deref'] = 'result_as_arg'
            elif result_typemap.cxx_type == 'std::string':
                result_as_string = ast.result_as_voidstar(
                    'stringout', result_name, const=ast.const)
                attrs = result_as_string.attrs
                attrs['context'] = options.C_var_context_template.format(c_var=result_name)
                self.move_arg_attributes(attrs, node, C_new)
            elif result_is_ptr:  # 'char *'
                result_as_string = ast.result_as_voidstar(
                    'charout', result_name, const=ast.const)
                attrs = result_as_string.attrs
                attrs['context'] = options.C_var_context_template.format(c_var=result_name)
                self.move_arg_attributes(attrs, node, C_new)
            else:  # char
                result_as_string = ast.result_as_arg(result_name)
                attrs = result_as_string.attrs
                attrs['len'] = options.C_var_len_template.format(c_var=result_name)
            attrs['intent'] = 'out'
            attrs['_is_result'] = True
            attrs['_generated_suffix'] = '_buf'
            # convert to subroutine
            C_new._subprogram = 'subroutine'
        elif has_allocatable_result:
            # Non-string and Non-char results
            self.setup_allocatable_result(C_new)

        if result_as_arg:
            # Create Fortran function without bufferify function_suffix but
            # with len attributes on string arguments.
            F_new = C_new.clone()
            ordered_functions.append(F_new)
            self.append_function_index(F_new)

            # Fortran function should wrap the new C function
            F_new._PTR_F_C_index = C_new._function_index
            options = F_new.options
            options.wrap_c = False
            options.wrap_fortran = True
            options.wrap_python = False
            options.wrap_lua = False
            # Do not add '_bufferify'
            F_new.fmtdict.function_suffix = node.fmtdict.function_suffix

            # Do not wrap original function (does not have result argument)
            node.options.wrap_fortran = False
        else:
            # Fortran function may call C subroutine if string result
            node._PTR_F_C_index = C_new._function_index

    def setup_allocatable_result(self, node):
        """node has a result with deref(allocatable).

        C wrapper:
           Add context argument for result
           Fill in values to describe array.

        Fortran:
            c_step1(context)
            allocate(Fout(len))
            c_step2(context, Fout, size(len))

        """
        options = node.options
        fmt_func = node.fmtdict
        attrs = node.ast.attrs

        # XXX - c_var is duplicated in wrapc.py wrap_function
        c_var = fmt_func.C_local + fmt_func.C_result
        attrs['context'] = options.C_var_context_template.format(
            c_var=c_var)

        node.statements = {}
        node.statements['c'] = dict(
            result_buf=dict(
                buf_args=['context'],
                c_helper='array_context copy_array',
                post_call=[
                    '{c_var_context}->cxx.addr  = {cxx_var};',
                    '{c_var_context}->cxx.idtor = {idtor};',
                    '{c_var_context}->addr.cvoidp = {cxx_var};',
                    '{c_var_context}->len = sizeof({cxx_type});',
                    '{c_var_context}->size = *{c_var_dimension};',
                ],
            ),
        )
        node.statements['f'] = dict(
            result_allocatable=dict(
                buf_args=['context'],
                f_helper='array_context copy_array_{cxx_type}',
                post_call=[
                    # XXX - allocate scalar
                    'allocate({f_var}({c_var_dimension}))',
                    'call SHROUD_copy_array_{cxx_type}'
                    '({c_var_context}, {f_var}, size({f_var}, kind=C_SIZE_T))',
                ],
            ),
        )

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
        node.F_module_dependencies = F_modules

    def XXXcheck_function_dependencies(self, node, used_types):
        """Record which types are used by a function.
        """
        if node.cxx_template:
            # The templated type will raise an error.
            # XXX - Maybe dummy it out
            # XXX - process templated types
            return
        ast = node.ast
        rv_type = ast.typename
        typedef = typemap.lookup_type(rv_type)
        if typedef is None:
            raise RuntimeError(
                "Unknown type {} for function decl: {}"
                .format(rv_type, node['decl']))
        result_typedef = typemap.lookup_type(rv_type)
        # XXX - make sure it exists
        used_types[rv_type] = result_typedef
        for arg in ast.params:
            argtype = arg.typename
            typedef = typemap.lookup_type(argtype)
            if typedef is None:
                raise RuntimeError("%s not defined" % argtype)
            if typedef.base == 'shadow':
                used_types[argtype] = typedef

    def gen_functions_decl(self, functions):
        """ Generate declgen for generated all functions.
        """
        for node in functions:
            node.declgen = node.ast.gen_decl()


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
    def __init__(self, newlibrary, config):
        self.newlibrary = newlibrary
        self.config = config

    def name_library(self):
        """entry pointer for library"""
        self.name_language(self.name_function_c)
        self.name_language(self.name_function_fortran)

    def name_language(self, handler):
        newlibrary = self.newlibrary
        for cls in newlibrary.classes:
            for func in cls.functions:
                handler(cls, func)

        for func in newlibrary.functions:
            handler(None, func)

    def name_function_c(self, cls, node):
        options = node.options
        if not options.wrap_c:
            return
        fmt_func = node.fmtdict

        node.eval_template('C_name')
        node.eval_template('F_C_name')
        fmt_func.F_C_name = fmt_func.F_C_name.lower()

    def name_function_fortran(self, cls, node):
        """ Must process C functions to generate their names.
        """
        options = node.options
        if not options.wrap_fortran:
            return

        node.eval_template('F_name_impl')
        node.eval_template('F_name_function')
        node.eval_template('F_name_generic')


class Preprocess(object):
    """Compute some state for functions."""
    def __init__(self, newlibrary, config):
        self.newlibrary = newlibrary
        self.config = config

    def process_library(self):
        """entry pointer for library"""
        newlibrary = self.newlibrary
        for cls in newlibrary.classes:
            for func in cls.functions:
                self.process_function(cls, func)

        for func in newlibrary.functions:
            self.process_function(None, func)

    def process_function(self, cls, node):
        options = node.options

        # Any nodes with cxx_template have been replaced with nodes
        # that have the template expanded.
        if not node.cxx_template:
            self.process_xxx(cls, node)
            self.check_pointer(node, node.ast)

    def process_xxx(self, cls, node):
        """Compute information common to all wrapper languages.

        Compute subprogram.  This may be different for each language.
        CXX_subprogram - The C++ function being wrapped.
        C_subprogram - functions will be converted to subroutines for
            return_this and destructors.
            A subroutine can be converted to a function by C_return_type.

        return_this = True for C++ functions which return 'this',
        are easier to call from Fortran if they are subroutines.
        There is no way to chain in Fortran:  obj->doA()->doB();

#        Lookup up typemap for result and arguments
        """

        fmt_func = node.fmtdict

        ast = node.ast
        CXX_result_type = ast.typename
        C_result_type = CXX_result_type
        F_result_type = CXX_result_type
        subprogram = ast.get_subprogram()
        node.CXX_subprogram = subprogram
        is_dtor = ast.attrs.get('_destructor', False)

        if node.return_this or is_dtor:
            CXX_result_type = 'void'
            C_result_type = 'void'
            F_result_type = 'void'
            node.CXX_subprogram = 'subroutine'
            subprogram = 'subroutine'
        elif fmt_func.C_custom_return_type:
            C_result_type = fmt_func.C_custom_return_type
            F_result_type = fmt_func.C_custom_return_type
            subprogram = 'function'

        node.C_subprogram = subprogram
        node.F_subprogram = subprogram

        node.CXX_return_type = CXX_result_type
        node.C_return_type = C_result_type
        node.F_return_type = F_result_type

        node.CXX_result_typemap = typemap.lookup_type(CXX_result_type)
        node.C_result_typemap = typemap.lookup_type(C_result_type)
        node.F_result_typemap = typemap.lookup_type(F_result_type)
#        if not result_typedef:
#            raise RuntimeError("Unknown type {} in {}",
#                               CXX_result_type, fmt_func.function_name)

    def check_pointer(self, node, ast):
        """Compute how to deal with a pointer function result.
        """
        options = node.options
        attrs = ast.attrs
        result_typemap = node.CXX_result_typemap
        ast.return_pointer_as = None
        if result_typemap.cxx_type == 'void':
            # subprogram == subroutine
            # deref may be set when a string function is converted into a subroutine.
            if 'deref' in attrs:
                ast.return_pointer_as = attrs['deref']
        elif result_typemap.base == 'shadow':
            # Change a C++ pointer into a Fortran pointer
            # return 'void *' as 'type(C_PTR)'
            # 'shadow' assigns pointer to type(C_PTR) in a derived type
            pass
        elif result_typemap.base == 'string':
            if 'deref' in attrs:
                ast.return_pointer_as = attrs['deref']
            else:
                # Default strings to create a Fortran allocatable.
                ast.return_pointer_as = 'allocatable'
        elif ast.is_indirect():
            # pointer to a POD  e.g. int *
            if 'deref' in attrs:
                ast.return_pointer_as = attrs['deref']
            elif 'dimension' in attrs:
                ast.return_pointer_as = 'pointer'
            elif options.return_scalar_pointer == 'pointer':
                ast.return_pointer_as = 'pointer'
            else:
                ast.return_pointer_as = 'scalar'
        else:
            if 'deref' in attrs:
                raise RuntimeError(
                    "Cannot have attribute 'deref' on non-pointer in {}"
                    .format(node.decl))

def generate_functions(library, config):
    VerifyAttrs(library, config).verify_attrs()
    GenFunctions(library, config).gen_library()
    Namify(library, config).name_library()
    Preprocess(library, config).process_library()

######################################################################

class CheckImplied(todict.PrintNode):
    """Check arguments in the implied attribute.
    """
    def __init__(self, expr, func):
        super(CheckImplied, self).__init__()
        self.expr = expr
        self.func = func

    def visit_Identifier(self, node):
        """Check arguments to size function.
        """
        if node.args == None:
            return node.name
        elif node.name == 'size':
            # size(arg)
            if len(node.args) != 1:
                raise RuntimeError("Too many arguments to 'size': "
                                   .format(self.expr))
            argname = node.args[0].name
            arg = self.func.ast.find_arg_by_name(argname)
            if arg is None:
                raise RuntimeError("Unknown argument '{}': {}"
                                   .format(argname, self.expr))
            if 'dimension' not in arg.attrs:
                raise RuntimeError(
                    "Argument '{}' must have dimension attribute: {}"
                    .format(argname, self.expr))
            return 'size'
        else:
            raise RuntimeError("Unexpected function '{}' in expression: {}"
                               .format(node.name, self.expr))

def check_implied(expr, func):
    """Check implied attribute expression for errors.
    """
    node = declast.ExprParser(expr).expression()
    visitor = CheckImplied(expr, func)
    return visitor.visit(node)
