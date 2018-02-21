.. Copyright (c) 2017-2018, Lawrence Livermore National Security, LLC. 
.. Produced at the Lawrence Livermore National Laboratory 
..
.. LLNL-CODE-738041.
.. All rights reserved. 
..
.. This file is part of Shroud.  For details, see
.. https://github.com/LLNL/shroud. Please also read shroud/LICENSE.
..
.. Redistribution and use in source and binary forms, with or without
.. modification, are permitted provided that the following conditions are
.. met:
..
.. * Redistributions of source code must retain the above copyright
..   notice, this list of conditions and the disclaimer below.
.. 
.. * Redistributions in binary form must reproduce the above copyright
..   notice, this list of conditions and the disclaimer (as noted below)
..   in the documentation and/or other materials provided with the
..   distribution.
..
.. * Neither the name of the LLNS/LLNL nor the names of its contributors
..   may be used to endorse or promote products derived from this
..   software without specific prior written permission.
..
.. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
.. "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
.. LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
.. A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL LAWRENCE
.. LIVERMORE NATIONAL SECURITY, LLC, THE U.S. DEPARTMENT OF ENERGY OR
.. CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
.. EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
.. PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
.. PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
.. LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
.. NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
.. SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
..
.. #######################################################################

Reference
=========

Command Line Options
--------------------

help
       Show this help message and exit.

version
       Show program's version number and exit.

outdir OUTDIR
       Directory for output files.
       Defaults to current directory.

outdir-c-fortran OUTDIR_C_FORTRAN
       Directory for C/Fortran wrapper output files, overrides *--outdir*.

outdir-python OUTDIR_PYTHON
       Directory for Python wrapper output files, overrides *--outdir*.

outdir-lua OUTDIR_LUA
       Directory for Lua wrapper output files, overrides *--outdir*.

outdir-yaml OUTDIR_YAML
       Directory for YAML output files, overrides *--outdir*.

logdir LOGDIR
       Directory for log files.
       Defaults to current directory.

cfiles CFILES
       Output file with list of C and C++ files created.

ffiles FFILES
       Output file with list of Fortran created.

path PATH
       Colon delimited paths to search for splicer files, may
       be supplied multiple times to append to path.

sitedir
       Return the installation directory of shroud and exit.
       This path can be used to find cmake/SetupShroud.cmake.

Global Fields
-------------

copyright
   A list of lines to add to the top of each generate file.
   Do not include any language specific comment characters since
   Shroud will add the appropriate comment delimiters for each language.

classes
  A list of classes.  Each class may have fields as detailed in 
  `Class Fields`_.

cxx_header
  Blank delimited list of header files which
  will be included in the implementation file.

format
   Dictionary of Format fields for the library.
   Described in `Format Fields`_.

language
  The language of the library to wrap.
  Valid values are ``c`` and ``c++``.
  The default is ``c++``.

library
  The name of the library.
  Used to name output files and modules.
  The first three letters are used as the default for **C_prefix** option.
  Defaults to *default_library*.
  Each YAML file is intended to wrap a single library.

namespace
  Blank delimited list of namespaces for **cxx_header**.
  The namespaces will be nested.

options
   Dictionary of option fields for the library.
   Described in `Options`_

patterns
   Code blocks to insert into generated code.
   Described in `Patterns`_.

splicer
   A dictionary mapping file suffix to a list of splicer files
   to read::

      splicer:
        c:
        -  filename1.c
        -  filename2.c

types
   A dictionary of user define types.
   Each type is a dictionary for members describing how to
   map a type between languages.
   Described in :ref:`TypesAnchor` and `Types Map`_.

.. _ClassFields:

Class Fields
------------

cxx_header
  C++ header file name which will be included in the implementation file.
  If unset then the global *cxx_header* will be used.

format
   Format fields for the class.
   Creates scope within library.
   Described in `Format Fields`_.

functions
   A list of functions in the class. Each function is defined by `Function Fields`_

options
   Options fields for the class.
   Creates scope within library.
   Described in `Options`_

namespace
  Blank delimited list of namespaces for **cxx_header**.
  The namespaces will be nested.
  If not defined then the global *namespace* will be used.
  If it starts with a ``-`` then no namespace will be used.


Function Fields
---------------

Each function can define fields to define the function
and how it should be wrapped.  These fields apply only
to a single function i.e. they are not inherited.

C_prototype
   XXX  override prototype of generated C function

cxx_template
   A dictionary of lists that define how each templated argument
   should be instantiated::

      decl: void Function7(ArgType arg)
      cxx_template:
        ArgType:
        - int
        - double

decl
   Function declaration.
   Parsed to extract function name, type and arguments descriptions.

default_arg_suffix
   A list of suffixes to apply to C and Fortran functions generated when
   wrapping a C++ function with default arguments.  The first entry is for
   the function with the fewest arguments and the final entry should be for
   all of the arguments.

format
   Format fields for the function.
   Creates scope within container (library or class).
   Described in `Format Fields`_.

fortran_generic
    A dictionary of lists that define generic functions which will be
    created.  This allows different types to be passed to the function.
    This feature is provided by C which will promote arguments::

      decl: void Function9(double arg)
      fortran_generic:
         arg:
         -  float
         -  double

options
   Options fields for the function.
   Creates scope within container (library or class).
   Described in `Options`_

return_this
   If true, the method returns a reference to ``this``.  This idiom can be used
   to chain calls in C++.  This idiom does not translate to C and Fortran.
   Instead the *C_return_type* format is set to ``void``.


Options
-------

debug
  Print additional comments in generated files that may 
  be useful for debugging.
  Defaults to *false*.

C_extern_C
   Set to *true* when the C++ routine is ``extern "C"``.
   Defaults to *false*.

C_line_length
  Control length of output line for generated C.
  This is not an exact line width, but is instead a hint of where
  to break lines.
  A value of 0 will give the shortest possible lines.
  Defaults to 72.

F_line_length
  Control length of output line for generated Fortran.
  This is not an exact line width, but is instead a hint of where
  to break lines.
  A value of 0 will give the shortest possible lines.
  Defaults to 72.

F_force_wrapper
  If *true*, always create an explicit Fortran wrapper.
  If *false*, only create the wrapper when there is work for it to do;
  otherwise, call the C function directly.
  For example, a function which only deals with native
  numeric types does not need a wrapper since it can be called
  directly by defining the correct interface.
  The default is *false*.

F_standard
  The fortran standard.  Defaults to *2003*.
  This effects the ``mold`` argument of the ``allocate`` statement.

F_string_len_trim
  For each function with a ``std::string`` argument, create another C
  function which accepts a buffer and length.  The C wrapper will call
  the ``std::string`` constructor, instead of the Fortran wrapper
  creating a ``NULL`` terminated string using ``trim``.  This avoids
  copying the string in the Fortran wrapper.
  Defaults to *true*.

.. bufferify

show_splicer_comments
    If ``true`` show comments which delineate the splicer blocks;
    else, do not show the comments.
    Only the global level option is used.

wrap_c
  If *true*, create C wrappers.
  Defaults to *true*.

wrap_fortran
  If *true*, create Fortran wrappers.
  Defaults to *true*.

wrap_python
  If *true*, create Python wrappers.
  Defaults to *false*.

wrap_lua
  If *true*, create Lua wrappers.
  Defaults to *false*.


Option Templates
^^^^^^^^^^^^^^^^

Templates are set in options then expanded to assign to the format 
dictionary.

C_header_filename_class_template
    ``wrap{cxx_class}.{C_header_filename_suffix}``

C_header_filename_library_template
   ``wrap{library}.{C_header_filename_suffix}``

C_impl_filename_class_template
    ``wrap{cxx_class}.{C_impl_filename_suffix}``

C_impl_filename_library_template
    ``wrap{library}.{C_impl_filename_suffix}``

C_name_template
    ``{C_prefix}{class_prefix}{underscore_name}{function_suffix}``

C_var_len_template
    Format for variable created with *len* annotation.
    Default ``N{c_var}``

C_var_size_template
    Format for variable created with *size* annotation.
    Default ``S{c_var}``

C_var_trim_template
    Format for variable created with *len_trim* annotation.
    Default ``L{c_var}``

class_prefix_template
    Class component for function names.
    Will be blank if the function is not in a class.
    ``{class_lower}_``

F_C_name_template
    ``{F_C_prefix}{class_prefix}{underscore_name}{function_suffix}``

F_abstract_interface_argument_template
   The name of arguments for an abstract interface used with function pointers.
   Defaults to ``{underscore_name}_{argname}``
   where *argname* is the name of the function argument.
   see :ref:`TypesAnchor_Function_Pointers`.

F_abstract_interface_subprogram_template
   The name of the abstract interface subprogram which represents a
   function pointer.
   Defaults to ``arg{index}`` where *index* is the 0-based argument index.
   see :ref:`TypesAnchor_Function_Pointers`.

F_name_generic_template
    ``{underscore_name}``

F_impl_filename_class_template
    ``wrapf{cxx_class}.{F_filename_suffix}``

F_impl_filename_library_template
    ``wrapf{library_lower}.{F_filename_suffix}``

F_name_impl_template
    ``{class_prefix}{underscore_name}{function_suffix}``

F_module_name_class_template
    ``{class_lower}_mod``

F_module_name_library_template
    ``{library_lower}_mod``

F_name_function_template
    ``{underscore_name}{function_suffix}``

LUA_class_reg_template
    Name of `luaL_Reg` array of function names for a class.
    ``{LUA_prefix}{cxx_class}_Reg``

LUA_ctor_name_template
    Name of constructor for a class.
    Added to the library's table.
    ``{cxx_class}``

LUA_header_filename_template
    ``lua{library}module.{LUA_header_filename_suffix}``

LUA_metadata_template
    Name of metatable for a class.
    ``{cxx_class}.metatable``

LUA_module_filename_template
    ``lua{library}module.{LUA_impl_filename_suffix}``

LUA_module_reg_template
    Name of `luaL_Reg` array of function names for a library.
    ``{LUA_prefix}{library}_Reg``

LUA_name_impl_template
    Name of implementation function.
    All overloaded function use the same Lua wrapper so 
    *function_suffix* is not needed.
    ``{LUA_prefix}{class_prefix}{underscore_name}``

LUA_name_template
    Name of function as know by Lua.
    All overloaded function use the same Lua wrapper so 
    *function_suffix* is not needed.
    ``{function_name}``

LUA_userdata_type_template
    ``{LUA_prefix}{cxx_class}_Type``

LUA_userdata_member_template
    Name of pointer to class instance in userdata.
    ``self``

YAML_type_filename_template
    Default value for global field YAML_type_filename
    ``{library_lower}_types.yaml``


Format Fields
-------------

Each scope (library, class, function) has its own format dictionary.
If a value is not found in the dictionary, then the parent
scope will be recursively searched.

Library
^^^^^^^

C_bufferify_suffix
  Suffix appended to generated routine which pass strings as buffers
  with explicit lengths.
  Defaults to *_bufferify*

C_header_filename
    Name of generated header file for the library.
    Defaulted from expansion of option *C_header_filename_library_template*.

C_header_filename_suffix:
   Suffix added to C header files.
   Defaults to ``h``.
   Other useful values might be ``hh`` or ``hxx``.

C_impl_filename
    Name of generated C++ implementation file for the library.
    Defaulted from expansion of option *C_impl_filename_library_template*.

C_impl_filename_suffix:
   Suffix added to C implementation files.
   Defaults to ``cpp``.
   Other useful values might be ``cc`` or ``cxx``.

C_local
    Prefix for C compatible local variable.
    Defaults to *SHC_*.

C_result
    The name of the C wrapper's result variable.
    It must not be the same as any of the routines arguments.
    It defaults to *rv*.

C_string_result_as_arg
    The name of the output argument for string results.
    Function which return ``char`` or ``std::string`` values return
    the result in an additional argument in the C wrapper.
    See also *F_string_result_as_arg*.

c_temp
    Prefix for wrapper temporary working variables.
    Defaults to *SHT_*.

C_this
    Name of the C object argument.  Defaults to ``self``.
    It may be necessary to set this if it conflicts with an argument name.

CXX_local
    Prefix for C++ compatible local variable.
    Defaults to *SHCXX_*.

CXX_this
    Name of the C++ object pointer set from the *C_this* argument.
    Defaults to ``SH_this``.

F_C_prefix
    Prefix added to name of generated Fortran interface for C routines.
    Defaults to **c_**.

F_derived_member
    The name of the member of the Fortran derived type which
    wraps a C++ class.  It will contain a ``type(C_PTR)`` which
    points to the C++ instance.
    Defaults to *voidptr*.

F_filename_suffix:
   Suffix added to Fortran files.
   Defaults to ``f``.
   Other useful values might be ``F`` or ``f90``.

F_module_name
    Name of module for Fortran interface for the library.
    Defaulted from expansion of option *F_module_name_library_template*
    which is **{library_lower}_mod**.

F_impl_filename
    Name of generated Fortran implementation file for the library.
    Defaulted from expansion of option *F_impl_filename_library_template*.
    If option *F_module_per_class* is false, then all derived types
    generated for each class will also be in this file.

F_result
    The name of the Fortran wrapper's result variable.
    It must not be the same as any of the routines arguments.
    It defaults to *SH_rv*  (Shroud return value).

F_string_result_as_arg
    The name of the output argument.
    Function which return a ``char *`` will instead by converted to a
    subroutine which require an additional argument for the result.
    See also *C_string_result_as_arg*.

F_this
   Name of the Fortran argument which is the derived type
   which represents a C++ class.
   It must not be the same as any of the routines arguments.
   Defaults to ``obj``.

library
    The value of global **field** *library*.

library_lower
    Lowercase version of *library*.

library_upper
    Uppercase version of *library*.

LUA_header_filename_suffix
   Suffix added to Lua header files.
   Defaults to ``h``.
   Other useful values might be ``hh`` or ``hxx``.

LUA_impl_filename_suffix
   Suffix added to Lua implementation files.
   Defaults to ``cpp``.
   Other useful values might be ``cc`` or ``cxx``.

LUA_module_name
    Name of Lua module for library.
    ``{library_lower}``

LUA_prefix
    Prefix added to Lua wrapper functions.

LUA_result
    The name of the Lua wrapper's result variable.
    It defaults to *rv*  (return value).

LUA_state_var
    Name of argument in Lua wrapper functions for lua_State pointer.

namespace_scope
    The values in field **namespace** delimited with ``::``.

PY_header_filename_suffix
   Suffix added to Python header files.
   Defaults to ``h``.
   Other useful values might be ``hh`` or ``hxx``.

PY_impl_filename_suffix
   Suffix added to Python implementation files.
   Defaults to ``cpp``.
   Other useful values might be ``cc`` or ``cxx``.

PY_module_name
    Name of wrapper Python module.
    Defaults to library name.

PY_name_impl
    Name of Python wrapper implemenation function.
    Defaults to *{PY_prefix}{class_prefix}{function_name}{function_suffix}*.

PY_prefix
    Prefix added to Python wrapper functions.

PY_result
    The name of the Python wrapper's result variable.
    It defaults to *SHTPy_rv*  (return value).

stdlib
    Name of C++ standard library prefix.
    blank when *language=c*.
    ``std::`` when *language=c++*.

YAML_type_filename
    Output filename for type maps for classes.

Class
^^^^^

C_header_filename
    Name of generated header file for the class.
    Defaulted from expansion of option *C_header_filename_class_template*.

C_impl_file
    Name of generated C++ implementation file for the library.
    Defaulted from expansion of option *C_impl_filename_class_template*.

F_derived_name
   Name of Fortran derived type for this class.
   Defaults to the C++ class name.

F_impl_filename
    Name of generated Fortran implementation file for the library.
    Defaulted from expansion of option *F_impl_filename_class_template*.
    Only defined if *F_module_per_class* is true.

F_module_name
    Name of module for Fortran interface for the class.
    Defaulted from expansion of option *F_module_name_class_template*
    which is **{class_lower}_mod**.
    Only defined if *F_module_per_class* is true.

F_name_associated
    Name of method to report if aa is associated.
    If the name is blank, no function is generated.

F_name_instance_get
    Name of method to get ``type(C_PTR)`` instance pointer from wrapped class.
    Defaults to *get_instance*.
    If the name is blank, no function is generated.

F_name_instance_set
    Name of method to set ``type(C_PTR)`` instance pointer in wrapped class.
    Defaults to *set_instance*.
    If the name is blank, no function is generated.

cxx_class
    The name of the C++ class from the YAML input file.

class_lower
    Lowercase version of *cxx_class*.

class_upper
    Uppercase version of *cxx_class*.

class_prefix
    Variable which may be used in creating function names.
    Defaults to evaluation of *class_prefix_template*.
    Outside of a class, set to empty string.

C_prefix
    Prefix for C wrapper functions.
    The prefix helps to ensure unique global names.
    Defaults to the first three letters of *library_upper*.


Function
^^^^^^^^

C_call_list
    Comma delimited list of function arguments.

.. uses tabs

C_call_code
    Code used to call function in C wrapper.

.. uses tabs

C_code
    User supplied wrapper code for the C wrapper for a function.

C_finalize
    User supplied code to perform any function finialization.
    Code added after all of the argument's *post_call* code.
    Can be used to free memory in the C wrapper.

.. evaluated in context of fmt_result

C_finalize_buf
    Identical to **C_finalize** but only applies to the buffer version of the
    wrapper routine.

C_name
    Name of the C wrapper function.
    Defaults to evaluation of option *C_name_template*.

C_post_call
    Statements added after the call to the function.
    Used to convert result and/or ``intent(OUT)`` arguments to C types.

.. C_post_call_pattern

C_pre_call
    Statements added before the call to the function.
    Used to convert C types to C++ types.

C_prototype
    C prototype for the function.
    This will include any arguments required by annotations or options,
    such as length or **F_string_result_as_arg**.  

.. uses tabs

C_return_code
    Code used to return from C wrapper.

C_return_type
    Return type of the function.
    If the **return_this** field is true, then set to ``void``.
    If the **C_return_type** format is set, use its value.
    Otherwise set to function's return type.

CXX_template
    The template component of the function declaration.
    ``<{type}>``

CXX_this_call
    How to call the function.
    ``{CXX_this}->`` for instance methods and blank for library functions.

F_arg_c_call
    Comma delimited arguments to call C function from Fortran.

.. uses tabs

F_arguments
    Set from option *F_arguments* or generated from YAML decl.

.. uses tabs

F_C_arguments
    Argument names to the ``bind(C)`` interface for the subprogram.

.. uses tabs

F_C_call
    The name of the C function to call.  Usually *F_C_name*, but it may
    be different if calling a generated routine.
    This can be done for functions with string arguments.

F_C_name
    Name of the Fortran ``BIND(C)`` interface for a C function.
    Defaults to the lower case version of *F_C_name_template*.

F_C_pure_clause
    TODO

F_C_result_clause
    Result clause for the ``bind(C)`` interface.

F_C_subprogram
    ``subroutine`` or ``function``.

F_call_code
    Code used to call function in Fortran wrapper.

.. uses tabs

F_code
    User supplied wrapper code for the Fortran wrapper for a function.

F_pure_clause
    For non-void function, ``pure`` if the *pure* annotation is added or 
    the function is ``const`` and all arguments are ``intent(in)``.

F_name_function
    The name of the *F_name_impl* subprogram when used as a
    type procedure.
    Defaults to evaluation of option *F_name_function_template*.

F_name_generic
    Defaults to evaluation of option *F_name_generic_template*.

F_name_impl
    Name of the Fortran implementation function.
    Defaults to evaluation of option *F_name_impl_template* .

F_result_clause
    `` result({F_result})`` for functions.
    Blank for subroutines.

function_name
    Name of function in the YAML file.

function_suffix
   Suffix to append to the end of generated name.

LUA_name
    Name of function as known by LUA.
    Defaults to evaluation of option *LUA_name_template*.

underscore_name
    *function_name* converted from CamelCase to snake_case.

function_suffix
    Suffix append to name.  Used to differentiate overloaded functions.
    Defaults to a sequence number (e.g. `_0`, `_1`, ...) but can be set
    by using the function field *function_suffix*.
    Multiple suffixes may be applied.

Argument
^^^^^^^^

c_const
    ``const`` if argument has the *const* attribute.

c_ptr
    `` * `` if argument is a pointer.

c_var
    The C name of the argument.

c_var_len
    Function argument generated from the *len* annotation.
    Used with char/string arguments.
    Set from option **C_var_len_template**.

c_var_size
    Function argument generated from the *size* annotation.
    Used with array/std::vector arguments.
    Set from option **C_var_size_template**.

c_var_trim
    Function argument generated from the *len_trim* annotation.
    Used with char/string arguments.
    Set from option **C_var_trim_template**.

cxx_deref
    Syntax to dereference argument.
    If *cxx_local_var* is *object*, then set to ``.``;
    if *pointer*, then set to ``->``.

cxx_T
    The template parameter for std::vector arguments.
    ``std::vector<cxx_T>``

cxx_type
    The C++ type of the argument.

cxx_var
    Name of the C++ variable.

f_var
    Fortran variable name for argument.


Result
------

cxx_rv_decl
    Declaration of variable to hold return value for function.



Types Map
---------

Types describe how to handle arguments from Fortran to C to C++.  Then
how to convert return values from C++ to C to Fortran.

Since Fortran 2003 (ISO/IEC 1539-1:2004(E)) there is a standardized
way to generate procedure and derived-type declarations and global
variables which are interoperable with C (ISO/IEC 9899:1999). The
bind(C) attribute has been added to inform the compiler that a symbol
shall be interoperable with C; also, some constraints are added. Note,
however, that not all C features have a Fortran equivalent or vice
versa. For instance, neither C's unsigned integers nor C's functions
with variable number of arguments have an equivalent in
Fortran. [#f1]_


.. list from util.py class Typedef

base
    Base type.
    For example, string and string_from_buffer both have a 
    base time of *string*.
    Defaults to *unknown*

forward
    Forward declaration.
    Defaults to *None*.

typedef
    Initialize from existing type
    Defaults to *None*.

c_header
    Name of C header file required for implementation.
    Only used with *language=c*.
    Defaults to *None*.

cxx_type
    Name of type in C++.
    Defaults to *None*.

cxx_to_c
    Expression to convert from C++ to C.
    Defaults to *None* which implies *{cxx_var}*.  i.e. no conversion required.

cxx_header
    Name of C++ header file required for implementation.
    For example, if cxx_to_c was a function.
    Only used with *language=c++*.
    Defaults to *None*.

c_type
    name of type in C.
    Defaults to *None*.

c_header
    Name of C header file required for type.
    This file is included in the interface header.
    Defaults to *None*.

c_to_cxx
    Expression to convert from C to C++.
    Defaults to *None* which implies *{c_var}*.  i.e. no conversion required.

c_statements
    A nested dictionary of code template to add.
    The first layer is *intent_in*, *intent_out*, *intent_inout*, *result*,
    *intent_in_buf*, *intent_out_buf*, *intent_inout_buf*, and *result_buf*.
    The second layer is *pre_call*, *pre_call_buf*, *post_call*, *cxx_header*.
    The entries are a list of format strings.

    intent_in
        Code to add for argument with ``intent(IN)``.
        Can be used to convert types or copy-in semantics.
        For example, ``char *`` to ``std::string``.

    intent_out
        Code to add after call when ``intent(OUT)``.
        Used to implement copy-out semantics.

    intent_inout
        Code to add after call when ``intent(INOUT)``.
        Used to implement copy-out semantics.

    result
        Code to use when passing result as an argument.


        buf_args
           An array of arguments which will be added to the
           bufferified version of a function.

           len
              Fortran intrinsic ``LEN``, of type *int*.

           len_trim
              Fortran intrinsic ``LEN_TRIM``, of type *int*.

           size
              Fortran intrinsic ``SIZE``, of type *long*.

        cxx_header
           string of blank delimited header names

        cxx_local_var
           Set if a local C++ variable is created.
           This is the case when C and C++ are not directly compatible.
           Usually a C++ constructor or cast is involved.
           Set to **scalar** when a local variable is being created, for example ``std::string``.
           Or set to **pointer** when used with a pointer, for example ``char *``.
           This sets *cxx_var* is set to ``SH_{c_var}``.

        c_helper
           A blank delimited list of helper routines to add.
           These functions are defined in whelper.py.
           There is no current way to add additional functions.

c_templates
    A dictionary indexed by type of specialized *c_statements*
    When an argument has a *template* field, such as type ``vector<string>``,
    some additional specialization of c_statements may be required::

        c_templates:
            string:
               intent_in_buf:
               - code to copy CHARACTER to vector<string>

f_c_args
    List of argument names to F_C routine.
    Defaults to *None*.

f_c_argdecl
    List of declarations to F_C routine.
    By default, only a single argument is passed for each dummy argument.
    Defaults to *None*.

f_c_module
    Fortran modules needed for type in the interface.
    A dictionary keyed on the module name with the value being a list of symbols.
    Similar to **f_module**.
    Defaults to *None*.

f_c_type
    Type declaration for ``bind(C)`` interface.
    Defaults to *None* which will then use *f_type*.

f_type
    Name of type in Fortran.
    Defaults to *None*.

f_derived_type
    Fortran derived type name.
    Defaults to *None* which will use the C++ class name
    for the Fortran derived type name.

.. f_args
    Arguments in the Fortran wrapper to pass to the C function.
    This can pass multiple arguments to C for a single
    argument to the wrapper; for example, an address and length
    for a ``character(*)`` argument.
    Or it may be intermediate values.
    For example, a Fortran character variable can be converted
    to a ``NULL`` terminated string with
    ``trim({var}) // C_NULL_CHAR``.
    Defaults to *None*  i.e. pass argument unchanged.

f_module
    Fortran modules needed for type in the implementation wrapper.
    A dictionary keyed on the module name with the value being a list of symbols.
    Defaults to *None*.::

        f_module:
           iso_c_binding:
             - C_INT

f_return_code
    Fortran code used to call function and assign the return value.
    Defaults to *None*.

f_cast
    Expression to convert Fortran type to C type.
    This is used when creating a Fortran generic functions which
    accept several type but call a single C function which expects
    a specific type.
    For example, type ``int`` is defined as ``int({f_var}, C_INT)``.
    This expression converts *f_var* to a ``integer(C_INT)``.
    Defaults to *{f_var}*  i.e. no conversion.

..  See tutorial function9 for example.  f_cast is only used if the types are different.

f_to_c
    Expression to convert Fortran type to C type.
    If this field is set, it will be used before f_cast.
    Defaults to *None*.

f_statement
    A nested dictionary of code template to add.
    The first layer is *intent_in*, *intent_out*, *intent_inout*, *result_pure* and *result*.
    The second layer is *declare*, *pre_call*, and *post_call*
    The entries are a list of format strings.

    c_local_var
        If true, generate a local variable using the C declaration for the argument.
        This variable can be used by the pre_call and post_call statements.
        A single declaration will be added even if with ``intent(inout)``.

    declare
        A list of declarations needed by *pre_call* or *f_post_call*.
        Usually a *c_local_var* is sufficient.
        If both *pre_call* and *post_call* are specified then both *declare*
        clause will be added and thus should not declare the same variable.

    pre_call
        Statement to execute before call, often to coerce types
        when *f_cast* cannot be used.

    call
        Code used to call the function.
        Defaults to ``{F_result} = {F_C_call}({F_arg_c_call})``

    post_call
        Statement to execute after call.
        Can be use to cleanup after *f_pre_call*
        or to coerce the return value.

    need_wrapper
        If true, the Fortran wrapper will always be created.
        This is used when an assignment is needed to do a type coercion;
        for example, with logical types.

..  XXX - maybe later.  For not in wrapping routines
..         f_attr_len_trim = None,
..         f_attr_len = None,
..         f_attr_size = None,

    f_helper
        Blank delimited list of helper function names to add to generated Fortran code.
        These functions are defined in whelper.py.
        There is no current way to add additional functions.

        private
           List of names which should be PRIVATE to the module

        interface
           Code to add to the non-executable part of the module.

        source
           Code to add in the CONTAINS section of the module.

result_as_arg
    Override fields when result should be treated as an argument.
    Defaults to *None*.

PY_build_arg
    Argument for Py_BuildValue.  Defaults to *{cxx_var}*.
    This field can be used to turn the argument into an expression such as
    *(int) {cxx_var}*  or *{cxx_var}{cxx_deref}c_str()*
    *PY_format* is used as the format:: 

       Py_BuildValue("{PY_format}", {PY_build_arg});

PY_format
    'format unit' for PyArg_Parse and Py_BuildValue.
    Defaults to *O*

PY_PyTypeObject
    Variable name of PyTypeObject instance.
    Defaults to *None*.

PY_PyObject
    Typedef name of PyObject instance.
    Defaults to *None*.

PY_ctor
    Expression to create object.
    ex. PyBool_FromLong({rv})
    Defaults to *None*.

PY_to_object
    PyBuild - object = converter(address).
    Defaults to *None*.

PY_from_object
    PyArg_Parse - status = converter(object, address).
    Defaults to *None*.

py_statement
    A nested dictionary of code template to add.
    The first layer is *intent_in*, *intent_out*, and *result*.
    The entries are a list of format strings.

..    declare
        A list of declarations needed by *pre_call* or *f_post_call*.

    post_parse
        Statements to execute after the call to ``PyArg_ParseTupleAndKeywords``.
        Used to convert C values into C++ values.
	Ex. ``{var} = PyObject_IsTrue({var_obj});``

    ctor
        Statements to create a Python object.
	Must ensure that ``py_var = cxx_var`` in some form.

..    post_call
        Statement to execute after call.
        Can be use to cleanup after *f_pre_call*
        or to coerce the return value.

        cxx_local_var
           True if a local C++ variable is created.
           This is the case when C and C++ are not directly compatible.
           Usually a C++ constructor or cast is involved.


Annotations
-----------

An annotation can be used to provide semantic information for a function or argument.

.. a.k.a. attributes

allocatable
   Adds the Fortran ``allocatable`` attribute to an argument and adds an
   ``allocate`` statement.
   see :ref:`TypesAnchor_Allocatable_array`.

default
   Default value for C++ function argument.
   This value is implied by C++ default argument syntax.

dimension
   Sets the Fortran DIMENSION attribute.
   Pointer argument should be passed through since it is an
   array.  *value* must be *False*
   If set without a value, it defaults to ``(*)``.

name
   Name of the method.
   Useful for constructor and destructor methods which have no names.

implied
   Used to compute value of argument to C++ based on argument
   to Fortran or Python wrapper.  Useful with array sizes::

       Sum(int * array +intent(in), int len +implied(size(array))

intent
   Valid valid values are ``in``, ``out``, ``inout``.
   If the argument is ``const``, the default is ``in``.

len
   For a string argument, pass an additional argument to the
   C wrapper with the result of the Fortran intrinsic ``len``.
   If a value for the attribute is provided it will be the name
   of the extra argument.  If no value is provided then the
   argument name defaults to option *C_var_len_template*.

   When used with a function, it will be the length of the return
   value of the function using the declaration::

     character(kind=C_CHAR, len={c_var_len}) :: {F_result}

len_trim
   For a string argument, pass an additional argument to the
   C wrapper with the result of the Fortran intrinsic ``len_trim``.
   If a value for the attribute is provided it will be the name
   of the extra argument.  If no value is provided then the
   argument name defaults to option *C_var_trim_template*.

pure
   Sets the Fortran PURE attribute.

value
   If true, pass-by-value; else, pass-by-reference.
   This attribute is implied when the argument is not a pointer or reference.


Doxygen
-------

Used to insert directives for doxygen for a function.

brief
   Brief description.

description
   Full description.

return
   Description of return value.


Patterns
--------

C_error_pattern
    Inserted after the call to the C++ function in the C wrapper.
    Format is evaluated in the context of the result argument.
    *c_var*, *c_var_len* refer to the result argument.

C_error_pattern_buf
    Inserted after the call to the C++ function in the buffer C wrapper
    for functions with string arguments.
    Format is evaluated in the context of the result argument.

PY_error_pattern
    Inserted into Python wrapper.


.. ......................................................................

.. rubric:: Footnotes

.. [#f1] https://gcc.gnu.org/onlinedocs/gfortran/Interoperability-with-C.html

