# Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and
# other Shroud Project Developers.
# See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

"""
Generating declarations

Options used to control generation:
   as_c        -
      references become pointers
   as_scalar
      Skip pointer declarations
   force_ptr
      change reference to pointer
   append_init
   ctor_dtor
   attrs
      Add attributes from YAML to the declaration.
      Must be False to create code that will compile.
   continuation

   lang - c_type or cxx_type

   in_params = True/False
   arg_lang = c_type or cxx_type
"""

class DeclStr(object):
    """
    Convert Declarator to a specific C declaration.

    A Declaration contains declarators
    """
    def __init__(self):
        super(DeclStr, self).__init__()
        self.reset()

    def reset(self):
        self.parts = []

        self.append_init = False
        self.attrs = False
        self.as_c = False
        self.as_ptr = False
        self.as_scalar = False
        self.continuation = False
        self.ctor_dtor = False
        self.force_ptr = False
        self.name = None
        self.add_params = True

        self.in_params = False
        self.arg_lang = None

        # From gen_arg_as_language
        self.lang = None
        self.asgn_value = False
        self.remove_const = False
        self.with_template_args = False
        return self

    def update(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        return self

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
            if self.in_params and self.arg_lang:
                # typedefs in C wrapper must use c_type typedef for arguments.
                # i.e. with function pointers
                parts.append(getattr(declaration.typemap, arg_lang))
            else:
                parts.append(" ".join(declaration.specifier))
        if declaration.template_arguments:
            parts.append(declaration.gen_template_arguments())

        self.declarator(declaration.declarator)
        #, decl, attrs=attrs,
#                        in_params=in_params, arg_lang=arg_lang,
#                        **kwargs)


    def declarator(self, declarator):
        """Generate string for Declarator.

        Appending text to self.parts.

        Replace name with value from self.name.
        name=None will skip appending any existing name.

        attrs=False give compilable code.
        """
        parts = self.parts
        if self.force_ptr:
            # Force to be a pointer
            parts.append(" *")
        elif self.as_scalar:
            pass  # Do not print pointer
        else:
            for ptr in declarator.pointer:
                self.ptr(ptr)
        if declarator.func:
            parts.append(" (")
            self.declarator(declarator.func)
            parts.append(")")
        elif self.name:
            if self.name:
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
        #        if use_attrs:
        #            declarator.gen_attrs(self.attrs, decl)

        if not self.add_params:
            pass
        elif declarator.params is not None:
            parts.append("(")
            if self.continuation:
                parts.append("\t")
            if declarator.params:
                comma = ""
                self.in_params = True
                for arg in declarator.params:
                    parts.append(comma)
                    self.declaration(arg)
#                    arg.gen_decl_work(decl, attrs=attrs, continuation=continuation,
#                                      in_params=True, arg_lang=arg_lang)
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
            self.gen_attrs(self.attrs, decl)


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

    def gen_arg_as_cxx(self, declaration):
        """Generate C++ declaration of variable.
        No parameters or attributes.

        Used to generate declarations in wrappers.
        """
        self.lang = "cxx_type"
        self.parts = []
        self.gen_arg_as_lang(declaration)
        return "".join(self.parts)

    def gen_arg_as_c(self, declaration, name=None):
        """Return a string of the unparsed declaration.
        """
        self.lang = "c_type"
        if name is not None:
            self.name = name
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
#        lang,
#        continuation=False,
#        asgn_value=False,
#        remove_const=False,
#        with_template_args=False,
#        force_ptr=False,
#        **kwargs
        """Generate an argument for the C wrapper.
        C++ types are converted to C types using typemap.

        Args:
            lang = c_type or cxx_type
            continuation - True - insert tabs to aid continuations.
                           Defaults to False.
            asgn_value - If True, make sure the value can be assigned
                         by removing const. Defaults to False.
            remove_const - Defaults to False.
            as_ptr - Change reference to pointer
            force_ptr - Change a scalar into a pointer
            as_scalar - Do not print Ptr
            params - if None, do not print function parameters.
            with_template_args - if True, print template arguments

        If a templated type, assume std::vector.
        The C argument will be a pointer to the template type.
        'std::vector<int> &'  generates 'int *'
        The length info is lost but will be provided as another argument
        to the C wrapper.
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
            typ = getattr(ntypemap, self.lang) or "--NOTYPE--"
            parts.append(typ)

        declarator = declaration.declarator
        if declaration.is_ctor and self.lang == "c_type":
            # The C wrapper wants a pointer to the type.
            force_ptr = True

        if self.asgn_value and const_index is not None and not declaration.declarator.is_indirect():
            # Remove 'const' so the variable can be assigned to.
            parts[const_index] = ""
        elif self.remove_const and const_index is not None:
            parts[const_index] = ""

        if self.lang == "c_type":
            self.as_c = True
            self.declarator(declarator)
#            declarator.gen_decl_work(decl, as_c=True, force_ptr=force_ptr,
#                                     append_init=False, ctor_dtor=True,
#                                     attrs=False, continuation=continuation, **kwargs)
        else:
            self.declarator(declarator)
#            declarator.gen_decl_work(decl, force_ptr=force_ptr,
#                                     append_init=False, ctor_dtor=True,
#                                     attrs=False, continuation=continuation, **kwargs)

######################################################################
# Create some instances to change defaults.
    

decl_str = DeclStr()
decl_str_noparams = DeclStr().update(add_params=False)

gen_arg_as_language = DeclStr().gen_arg_as_language

gen_arg_instance = DeclStr().update(
    add_params=False,
    continuation=True
)
gen_arg_as_c = gen_arg_instance.gen_arg_as_c
gen_arg_as_cxx = gen_arg_instance.gen_arg_as_cxx
