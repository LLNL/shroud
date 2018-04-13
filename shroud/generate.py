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
            for func in cls.functions:
                self.check_fcn_attrs(func)

        for func in newlibrary.functions:
            self.check_fcn_attrs(func)

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
                    'allocatable',
                    'dimension',
                    'len',
                    'name',
                    'pure',
                    ]:
                raise RuntimeError(
                    "Illegal attribute '{}' for function {} in {}"
                    .format(attr, node.ast.name, node.decl))

        for arg in ast.params:
            self.check_arg_attrs(node, arg)

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
                    'dimension',
                    'hidden',  # omitted in Fortran API, returned from C++
                    'implied', # omitted in Fortran API, value passed to C++
                    'intent',
                    'len', 'len_trim', 'size',
                    'template',
                    'value',
                    ]:
                raise RuntimeError(
                    "Illegal attribute '{}' for argument {} in {}"
                    .format(attr, argname, node.decl))

        argtype = arg.typename
        typedef = typemap.Typedef.lookup(argtype)
        if typedef is None:
            # if the type does not exist, make sure it is defined by cxx_template
            #- decl: void Function7(ArgType arg)
            #  cxx_template:
            #    ArgType:
            #    - int
            #    - double
            if argtype not in node.cxx_template:
                raise RuntimeError("check_arg_attrs: No such type %s: %s" % (
                        argtype, node.decl))

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
        elif typedef and typedef.base == 'vector':
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
    Generate Typedef from class.
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

        self.function_index = newlibrary.function_index

        for cls in newlibrary.classes:
#            added = self.default_ctor_and_dtor(cls)
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

    def default_ctor_and_dtor(self, cls):
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
        """
        if len(node.cxx_template) != 1:
            # In the future it may be useful to have multiple templates
            # That the would start creating more permutations
            raise NotImplementedError("Only one cxx_templated type for now")
        for typename, types in node.cxx_template.items():
            for type in types:
                new = node.clone()
                ordered_functions.append(new)
                self.append_function_index(new)

                new._generated = 'cxx_template'
                fmt = new.fmtdict
                fmt.function_suffix = fmt.function_suffix + '_' + type
                new.cxx_template = {}
                options = new.options
                options.wrap_c = True
                options.wrap_fortran = True
                options.wrap_python = False
                options.wrap_lua = False
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
        """
        if len(node.fortran_generic) != 1:
            # In the future it may be useful to have multiple generic arguments
            # That the would start creating more permutations
            raise NotImplemented("Only one generic arg for now")
        for argname, types in node.fortran_generic.items():
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

    def arg_to_buffer(self, node, ordered_functions):
        """Look for function which have implied arguments.
        This includes functions with string or vector arguments.
        If found then create a new C function that
        will convert argument into a buffer and length.
        """
        options = node.options
        fmt = node.fmtdict

        # If a C++ function returns a std::string instance,
        # the default wrapper will not compile since the wrapper
        # will be declared as char. It will also want to return the
        # c_str of a stack variable. Warn and turn off the wrapper.
        ast = node.ast
        result_type = ast.typename
        result_typedef = typemap.Typedef.lookup(result_type)
        # shadow classes have not been added yet.
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
                                  .format(result_typedef.cxx_type,
                                          ast.name))

        if options.wrap_fortran is False:
            return
        if options.F_string_len_trim is False:  # XXX what about vector
            return

        # Is result or any argument a string or vector?
        # If so, additional arguments will be passed down so
        # create buffer version of function.
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
        is_pure = ast.attrs.get('pure', False)
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
        C_new = node.clone()
        ordered_functions.append(C_new)
        self.append_function_index(C_new)

        C_new._generated = 'arg_to_buffer'
        fmt = C_new.fmtdict
        fmt.function_suffix = fmt.function_suffix + fmt.C_bufferify_suffix

        options = C_new.options
        options.wrap_c = True
        options.wrap_fortran = False
        options.wrap_python = False
        options.wrap_lua = False
        C_new._PTR_C_CXX_index = node._function_index

        newargs = []
        for arg in C_new.ast.params:
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

        if has_string_result:
            # Add additional argument to hold result
            ast = C_new.ast
            if ast.attrs.get('allocatable', False):
                result_as_string = ast.result_as_voidstarstar(
                    'stringout', result_name, const=ast.const)
                attrs = result_as_string.attrs
                attrs['lenout'] = options.C_var_len_template.format(c_var=result_name)
            else:
                result_as_string = ast.result_as_arg(result_name)
                attrs = result_as_string.attrs
                attrs['len'] = options.C_var_len_template.format(c_var=result_name)
            attrs['intent'] = 'out'
            attrs['_is_result'] = True
            # convert to subroutine
            C_new._subprogram = 'subroutine'

        if is_pure:
            # pure functions which return a string have result_pure defined.
            pass
        elif result_as_arg:
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
        """entry pointer for class"""
        self.name_language(self.name_function_c)
        self.name_language(self.name_function_fortran)

    def name_language(self, handler):
        newlibrary = self.newlibrary
        for cls in newlibrary.classes:
            for func in cls.functions:
                handler(cls, func)

            options = cls.options
            fmt_class = cls.fmtdict

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
        fmt_func = node.fmtdict

        node.eval_template('F_name_impl')
        node.eval_template('F_name_function')
        node.eval_template('F_name_generic')


def generate_functions(library, config):
    VerifyAttrs(library, config).verify_attrs()
    GenFunctions(library, config).gen_library()
    Namify(library, config).name_library()

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
