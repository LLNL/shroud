# Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and
# other Shroud Project Developers.
# See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

"""
Generating declarations for wrappers from AST.
"""

from . import todict

class DeclStr(object):
    """
    Convert Declarator to a specific C declaration.

    A Declaration contains declarators

    add_params   - if False, do not print function parameters.
    append_init  -
    arg_lang     - c_type or cxx_type
    attrs        - Add attributes from YAML to the declaration.
                   Must be False to create code that will compile.
    continuation - True - insert tabs to aid continuations.
                   Defaults to False.
    ctor_dtor    -
    name         - False, do not add a name (abstract declarator).
                   non-None, use argument instead of declarator name.

    Internal state:
    as_c         - references become pointers.
    as_ptr       - Change reference to pointer
    force_ptr    - Change a scalar into a pointer
    in_params    - True/False
    lang         - c_type or cxx_type, field in Typemap.
    remove_const - Remove any const in declaration. Defaults to False.
    with_template_args - if True, print template arguments

    If a templated type, assume std::vector.
    The C argument will be a pointer to the template type.
    'std::vector<int> &'  generates 'int *'
    The length info is lost but will be provided as another argument
    to the C wrapper.

    """
    def __init__(self,
                 add_params=True,
                 append_init=False,
                 arg_lang=None,
                 attrs=False,
                 continuation=False,
                 ctor_dtor=False,
                 name=None):
        self.add_params = add_params
        self.append_init = append_init
        self.arg_lang = arg_lang
        self.attrs = attrs
        self.continuation = continuation
        self.ctor_dtor = ctor_dtor
        self.name = name

        self.parts = []
        self.as_c = False
        self.as_ptr = False
        self.force_ptr = False
        self.in_params = False
        # From gen_arg_as_language
        self.lang = None
        self.remove_const = False
        self.with_template_args = False

    def gen_decl(self, declaration):
        """Return a string of the unparsed declaration.

        Args:
            params - None do not print parameters.
        """
        self.parts = []
        self.declaration(declaration)
        return "".join(self.parts)

    def declaration(self, declaration):
        #, decl, attrs=True,
        #            in_params=False, arg_lang=None,
        #            **kwargs):
        """Generate string for Declaration.

        Append text to decl list.

        Replace params with value from kwargs.
        Most useful to call with params=None to skip parameters
        and only get function result.
        """
        parts = self.parts
        if declaration.const:
            parts.append("const ")

        if declaration.is_dtor:
            parts.append("~")
            parts.append(declaration.is_dtor)
        else:
            if declaration.storage:
                parts.append(" ".join(declaration.storage))
                parts.append(" ")

            ntypemap = declaration.typemap
            if ntypemap.is_enum and ntypemap.typedef and self.arg_lang:
                ntypemap = ntypemap.typedef
                parts.append(getattr(ntypemap, self.arg_lang))
            elif self.in_params and self.arg_lang:
                # typedefs in C wrapper must use c_type typedef for arguments.
                # i.e. with function pointers
                parts.append(getattr(ntypemap, self.arg_lang))
            else:
                parts.append(" ".join(declaration.specifier))
        if declaration.template_arguments:
            parts.append(declaration.gen_template_arguments())

        self.declarator(declaration.declarator)

    def declarator(self, declarator):
        """Generate string for Declarator.

        Appending text to self.parts.

        Replace name with value from self.name.
        name=None will skip appending any existing name.

        attrs=False give compilable code.
        """
        parts = self.parts
        if self.force_ptr:
            # Force to be a pointer, even if scalar
            parts.append(" *")
        else:
            for ptr in declarator.pointer:
                self.ptr(ptr)
        if declarator.func:
            parts.append(" (")
            self.declarator(declarator.func)
            parts.append(")")
        elif self.name is False:
            pass
        elif self.name is not None:
            parts.append(" ")
            parts.append(self.name)
        elif declarator.name:
            parts.append(" ")
            parts.append(declarator.name)
        elif self.ctor_dtor and declarator.ctor_dtor_name:
            parts.append(" ")
            parts.append(declarator.ctor_dtor_name)

        if self.append_init and declarator.init is not None:
            parts.append("=")
            parts.append(str(declarator.init))

        if self.add_params is False:
            pass
        elif declarator.params is not None:
            parts.append("(")
            if self.continuation:
                parts.append("\t")
            if declarator.params:
                comma = ""
                self.in_params = True
                self.name = None  # Do not override parameter names
                for arg in declarator.params:
                    parts.append(comma)
                    self.declaration(arg)
                    if self.continuation:
                        comma = ",\t "
                    else:
                        comma = ", "
                self.in_params = False
            else:
                parts.append("void")
            parts.append(")")
            if declarator.func_const:
                parts.append(" const")
        for dim in declarator.array:
            parts.append("[")
            parts.append(todict.print_node(dim))
            parts.append("]")
        if self.attrs:
            self.gen_attrs(declarator.attrs, parts)

    def gen_attrs(self, attrs, parts):
        space = " "
        for attr in sorted(attrs):
            if attr[0] == "_":  # internal attribute, __line__
                continue
            value = attrs[attr]
            parts.append(space)
            parts.append("+")
            if value is True:
                parts.append(attr)
            else:
                parts.append("{}({})".format(attr, value))
            space = ""

    def ptr(self, pointer):
        """Generate string by appending text to decl.
        """
        parts = self.parts
        if pointer.ptr:
            parts.append(" ")
            if self.as_c:
                # references become pointers with as_c
                parts.append("*")
            elif self.as_ptr:
                # Change reference to pointer
                parts.append("*")
            else:
                parts.append(pointer.ptr)
        if pointer.const:
            parts.append(" const")
        if pointer.volatile:
            parts.append(" volatile")

    ##########

    def gen_arg_as_cxx(self, declaration, add_params=True,
                       as_ptr=False, name=None,
                       force_ptr=False, remove_const=False,
                       with_template_args=False):
        """Generate C++ declaration of variable.
        No parameters or attributes.

        Used to generate declarations in wrappers implementation.
        """
        self.add_params = add_params
        self.as_ptr = as_ptr
        self.name = name
        self.force_ptr = force_ptr
        self.remove_const = remove_const
        self.with_template_args = with_template_args
        
        self.lang = "cxx_type"
        self.parts = []
        self.gen_arg_as_lang(declaration)
        return "".join(self.parts)

    def gen_arg_as_c(self, declaration, add_params=True, name=None,
                     remove_const=False, lang="c_type"):
        """Return a string of the unparsed declaration.

        Used to generate declarations in wrappers headers.

        lang = "c_type" or "ci_type"
        """
        self.add_params = add_params
        self.as_ptr = False
        self.name = name
        self.force_ptr = False
        self.remove_const = remove_const
        self.with_template_args = False
        
        self.lang = lang
        self.parts = []
        self.gen_arg_as_lang(declaration)
        return "".join(self.parts)

    def gen_arg_as_language(self, declaration, lang, name):
        """Generate C++ declaration of variable.
        No parameters or attributes.

        Called from add_var_getter_setter.

        Parameters
        ----------
        lang : str
            "c_type" or "cxx_type"
        """
        self.lang = lang
        self.name = name
        self.parts = []
        self.gen_arg_as_lang(declaration)
        return "".join(self.parts)

    def gen_arg_as_lang(self, declaration):
        """Generate an argument for the C wrapper.
        C++ types are converted to C types using typemap.

        Args:
            declaraton - declast.Declaration
        """
        parts = self.parts
        const_index = None
        if declaration.const:
            const_index = len(parts)
            parts.append("const ")

        if self.with_template_args and declaration.template_arguments:
            # Use template arguments from declaration
            typ = getattr(declaration.typemap, self.lang)
            if declaration.typemap.sgroup == "vector":
                # Vector types are not explicitly instantiated in the YAML file.
                parts.append(declaration.typemap.name)
                parts.append(declaration.gen_template_arguments())
            else:
                # cxx_type includes template  ex. user<int>
                parts.append(declaration.typemap.cxx_type)
        else:
            # Convert template_argument.
            # ex vector<int> -> int
            if declaration.template_arguments:
                ntypemap = declaration.template_arguments[0].typemap
            else:
                ntypemap = declaration.typemap
            if self.lang == "c_type":
                typ = ntypemap.c_type or "--NOTYPE--"
            elif self.lang == "cxx_type":
                typ = ntypemap.cxx_type or "--NOTYPE--"
            elif self.lang == "ci_type":
                typ = ntypemap.ci_type or ntypemap.c_type or "--NOTYPE--"
            else:
                typ = "--NOTYPE--"
            parts.append(typ)

        declarator = declaration.declarator

        if self.remove_const and const_index is not None:
            parts[const_index] = ""

        if self.lang == "cxx_type":
            self.as_c = False
            self.declarator(declarator)
        else:  # c_type and ci_type
            self.as_c = True
            # The C wrapper wants a pointer to the type.
            if declaration.is_ctor:
                self.force_ptr = True
            self.declarator(declarator)

######################################################################
# Create some instances to change defaults.
    

gen_decl = DeclStr(append_init=True, attrs=True).gen_decl
gen_decl_noparams = DeclStr(add_params=False, attrs=True).gen_decl

gen_arg_as_language = DeclStr().gen_arg_as_language

gen_arg_instance = DeclStr(
    append_init=False,
    ctor_dtor=True,
    attrs=False,
    continuation=True
)
gen_arg_as_c = gen_arg_instance.gen_arg_as_c
gen_arg_as_cxx = gen_arg_instance.gen_arg_as_cxx
