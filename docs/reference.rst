.. Copyright (c) 2017-2020, Lawrence Livermore National Security, LLC and
   other Shroud Project Developers.
   See the top-level COPYRIGHT file for details.

   SPDX-License-Identifier: (BSD-3-Clause)

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

write-helpers BASE
       Write files which contain the available helper functions
       into the files BASE.c and BASE.f.

write-version
       Write Shroud version into generated files.
       ``--nowrite-version`` will not write the version and is used
       by the testsuite to avoid changing every reference file when
       the version changes.

yaml-types FILE
       Write a YAML file with the default types.


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
  Defaults to *library*.
  Each YAML file is intended to wrap a single library.

options
   Dictionary of option fields for the library.
   Described in `Options`_

patterns
   Code blocks to insert into generated code.
   Described in `Patterns`_.

splicer
   A dictionary mapping file suffix to a list of splicer files
   to read:

.. code-block:: yaml

      splicer:
        c:
        -  filename1.c
        -  filename2.c

types
   A dictionary of user define types.
   Each type is a dictionary of members describing how to
   map a type between languages.
   Described in :ref:`TypemapsAnchor` and `Types Map`_.

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

declarations
   A list of declarations in the class.
   Each function is defined by `Function Fields`_

fields:
   A dictionary of fields used to update the typemap.

options
   Options fields for the class.
   Creates scope within library.
   Described in `Options`_


Function Fields
---------------

Each function can define fields to define the function
and how it should be wrapped.  These fields apply only
to a single function i.e. they are not inherited.

C_prototype
   XXX  override prototype of generated C function

cxx_template
   A list that define how each templated argument
   should be instantiated:

.. code-block:: yaml

      decl: void Function7(ArgType arg)
      cxx_template:
      - instantiation: <int>
      - instantiation: <double>

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
    This feature is provided by C which will promote arguments.
    Each generic function will have a suffix which defaults to an underscore
    plus a sequence number.
    This change be changed by adding *function_suffix* for a declaration.

.. code-block:: yaml

      decl: void GenericReal(double arg)
      fortran_generic:
      - decl: (float arg)
        function_suffix: suffix1
      - decl: (double arg)

    A full example is at :ref:`GenericReal <example_GenericReal>`.

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

C_API_case
   Control case of *C_name_scope*.
   Possible values are 'lower' or 'upper'.
   Any other value will have no effect.

C_extern_C
   Set to *true* when the C++ routine is ``extern "C"``.
   Defaults to *false*.

C_force_wrapper
  If *true*, always create an explicit C wrapper.
  When *language* is c++ a C wrapper is always created.
  When wrapping C, the wrapper is automatically created if there is work for it to do.
  For example, pre_call or post_call is defined.
  The user should set this option when wrapping C and the function is really
  a macro or a function pointer variable. This forces a function to be created
  allowing Fortran to use the macro or function pointer.

C_line_length
  Control length of output line for generated C.
  This is not an exact line width, but is instead a hint of where
  to break lines.
  A value of 0 will give the shortest possible lines.
  Defaults to 72.

class_ctor
  Indicates that this function is a constructor for a struct.
  The value is the name of the struct.
  Useful for *wrap_struct_as=class* when used with C.

.. code-block:: yaml

    - decl: struct Cstruct_as_class {
              int x1;
              int y1;
            };
      options:
        wrap_struct_as: class

    - decl: Cstruct_as_class *Create_Cstruct_as_class(void)
      options:
        class_ctor: Cstruct_as_class

CXX_standard
  C++ standard. Defaults to *2011*.
  See *nullptr*.

debug
  Print additional comments in generated files that may 
  be useful for debugging.
  Defaults to *false*.

debug_index
  Print index number of function and relationships between 
  C and Fortran wrappers in the wrappers and json file.
  The number changes whenever a new function
  is inserted and introduces lots of meaningless differenences in the test
  answers. This option is used to avoid the clutter.  If needed for 
  debugging, then set to *true*.  **debug** must also be *true*.
  Defaults to *false*.

doxygen
  If True, create doxygen comments.

F_create_bufferify_function
  Controls creation of a *bufferify* function.
  If *true*, an additional C function is created which receives
  *bufferified* arguments - i.e. the len, len_trim, and size may be
  added as additional arguments.  Set to *false* when when you want to
  avoid passing this information.  This will avoid a copy of
  ``CHARACTER`` arguments required to append a trailing null.
  Defaults to *true*.

F_create_generic
  Controls creation of a generic interface.  It defaults to *true* for
  most cases but will be set to *False* if a function is templated on
  the return type since Fortran does not distiuguish generics based on
  return type (similar to overloaded functions based on return type in
  C++).

.. XXX should also be set to false when the templated argument in
   cxx_template is part of the implementation and not the interface.

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

F_return_fortran_pointer
  Use ``c_f_pointer`` in the Fortran wrapper to return 
  a Fortran pointer instead of a ``type(C_PTR)``
  in routines which return a pointer
  It does not apply to ``char *``, ``void *``, and routines which return
  a pointer to a class instance.
  Defaults to *true*.

.. XXX how to decide length of pointer

literalinclude

  Write some text lines which can be used with Sphinx's literalinclude
  directive.  This is used to insert the generated code into the
  documentation.
  Can be applied at the top level or any declaration.
  Setting *literalinclude* at the top level implies *literalinclude2*.

literalinclude2

  Write some text lines which can be used with Sphinx's literalinclude
  directive.  Only effects some entities which do not map to a 
  declarations such as some helper functions or types.
  Only effective at the top level.

  Each Fortran interface will be encluded in its own ``interface`` block.
  This is to provide the interface context when code is added to the
  documentation.

PY_create_generic
  Controls creation of a multi-dispatch function with
  overloaded/templated functions.
  It defaults to *true* for
  most cases but will be set to *False* if a function is templated on
  the return type since Fortran does not distiuguish generics based on
  return type (similar to overloaded functions based on return type in
  C++).

.. XXX should also be set to false when the templated argument in
   cxx_template is part of the implementation and not the interface.

PY_write_helper_in_util
   When *True* helper functions will be written into the utility file
   *PY_utility_filename*. Useful when there are lots of classes since
   helper functions may be duplicated in several files.
   The value of format *PY_helper_prefix* will have *C_prefix* append
   to create names that are unique to the library.
   Defaults to *False*.
   
return_scalar_pointer
  Determines how to treat a function which returns a pointer to a scalar
  (it does not have the *dimension* or *rank* attribute).
  **scalar** return as a scalar or **pointer** to return as a pointer.
  This option does not effect the C or Fortran wrapper.
  For Python, **pointer** will return a NumPy scalar.
  Defaults to *pointer*.

.. default_attr_deref
  
.. bufferify

show_splicer_comments
    If ``true`` show comments which delineate the splicer blocks;
    else, do not show the comments.
    Only the global level option is used.

wrap_class_as
    Defines how a ``class`` should be wrapped.
    If *class*, wrap using a shadow type.
    If *struct*, wrap the same as a ``struct``.
    Default is *class*.

wrap_struct_as
    Defines how a ``struct`` should be wrapped.
    If *struct*, wrap a struct as a Fortran derived-type.
    If *class*, wrap a struct the same as a class using a shadow type.
    Default is *struct*.
    
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

C_enum_template
    Name of enumeration in C wrapper.
    ``{C_prefix}{C_name_scope}{enum_name}``

C_enum_member_template
    Name of enumeration member in C wrapper.
    ``{C_prefix}{C_name_scope}{enum_member_name}``

C_header_filename_class_template
    ``wrap{file_scope}.{C_header_filename_suffix}``

C_header_filename_library_template
   ``wrap{library}.{C_header_filename_suffix}``

C_header_filename_namespace_template
   ``wrap{scope_file}.{C_header_filename_suffix}``

C_impl_filename_class_template
    ``wrap{file_scope}.{C_impl_filename_suffix}``

C_impl_filename_library_template
    ``wrap{library}.{C_impl_filename_suffix}``

C_impl_filename_namespace_template
    ``wrap{scope_file}.{C_impl_filename_suffix}``

C_memory_dtor_function_template
    Name of function used to delete memory allocated by C or C++.
    defaults to ``{C_prefix}SHROUD_memory_destructor``.

C_name_template
    ``{C_prefix}{C_name_scope}{underscore_name}{function_suffix}{template_suffix}``

C_var_len_template
    Format for variable created with *len* annotation.
    Default ``N{c_var}``

C_var_size_template
    Format for variable created with *size* annotation.
    Default ``S{c_var}``

C_var_trim_template
    Format for variable created with *len_trim* annotation.
    Default ``L{c_var}``

F_C_name_template
    ``{F_C_prefix}{F_name_scope}{underscore_name}{function_suffix}{template_suffix}``

F_abstract_interface_argument_template
   The name of arguments for an abstract interface used with function pointers.
   Defaults to ``{underscore_name}_{argname}``
   where *argname* is the name of the function argument.
   see :ref:`DeclAnchor_Function_Pointers`.

F_abstract_interface_subprogram_template
   The name of the abstract interface subprogram which represents a
   function pointer.
   Defaults to ``arg{index}`` where *index* is the 0-based argument index.
   see :ref:`DeclAnchor_Function_Pointers`.

F_array_type_template
   ``{C_prefix}SHROUD_array``
   
F_capsule_data_type_template
    Name of the derived type which is the ``BIND(C)`` equivalent of the
    struct used to implement a shadow class (**C_capsule_data_type**).
    All classes use the same derived type.
    Defaults to ``{C_prefix}SHROUD_capsule_data``.

F_capsule_type_template
    ``{C_prefix}SHROUD_capsule``
  

F_enum_member_template
    Name of enumeration member in Fortran wrapper.
    ``{F_name_scope}{enum_member_lower}``
    Note that *F_enum_template* does not exist since only the members are 
    in the Fortran code, not the enum name itself.

F_name_generic_template
    ``{underscore_name}``

F_impl_filename_library_template
    ``wrapf{library_lower}.{F_filename_suffix}``

F_name_impl_template
    ``{F_name_scope}{underscore_name}{function_suffix}{template_suffix}``

F_module_name_library_template
    ``{library_lower}_mod``

F_module_name_namespace_template
    ``{file_scope}_mod``

F_name_function_template
    ``{underscore_name}{function_suffix}{template_suffix}``

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
    ``{LUA_prefix}{C_name_scope}{underscore_name}``

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

PY_array_arg
    How to wrap arrays - numpy or list.
    Applies to function arguments and to structs when
    **PY_struct_arg** is *class* (struct-as-class).
    Defaults to *numpy*.
    Added to fmt for functions.
    Useful for *c_helpers* in statements.

.. code-block:: text

        c_helper="get_from_object_{c_type}_{PY_array_arg}",

PY_module_filename_template
    ``py{library}module.{PY_impl_filename_suffix}``

PY_header_filename_template
    ``py{library}module.{PY_header_filename_suffix}``

PY_utility_filename_template
    ``py{library}util.{PY_impl_filename_suffix}``

PY_PyTypeObject_template
    ``{PY_prefix}{cxx_class}_Type``

PY_PyObject_template
    ``{PY_prefix}{cxx_class}``

PY_member_getter_template
    Name of descriptor getter method for a class variable.
    ``{PY_prefix}{cxx_class}_{variable_name}_getter``

PY_member_setter_template
    Name of descriptor setter method for a class variable.
    ``{PY_prefix}{cxx_class}_{variable_name}_setter``

PY_member_object_template
    Name of struct member of type `PyObject *` which
    contains the data for member pointer fields.
    ``{variable_name}_obj``.

PY_name_impl_template
    ``{PY_prefix}{function_name}{function_suffix}{template_suffix}``

PY_numpy_array_capsule_name_template
    Name of ``PyCapsule object`` used as base object of NumPy arrays.
    Used to make sure a valid capsule is passed to *PY_numpy_array_dtor_function*.
    ``{PY_prefix}array_dtor``

PY_numpy_array_dtor_context_template
    Name of ``const char * []`` array used as the *context* field
    for *PY_numpy_array_dtor_function*.
    ``{PY_prefix}array_destructor_context``

PY_numpy_array_dtor_function_template
    Name of *destructor* in ``PyCapsule`` base object of NumPy arrays.
    ``{PY_prefix}array_destructor_function``

PY_struct_array_descr_create_template
    Name of C/C++ function to create a ``PyArray_Descr`` pointer for a structure.
    ``{PY_prefix}{cxx_class}_create_array_descr``

PY_struct_arg
    How to wrap structs - numpy, list or class.
    Defaults to *numpy*.

PY_struct_array_descr_variable_template
    Name of C/C++ variable which is a pointer to a ``PyArray_Descr``
    variable for a structure.
    ``{PY_prefix}{cxx_class}_array_descr``

PY_struct_array_descr_name_template
    Name of Python variable which is a ``numpy.dtype`` for a struct.
    Can be used to create instances of a C/C++ struct from Python.
    ``np.array((1,3.14), dtype=tutorial.struct1_dtype)``
    ``{cxx_class}_dtype``


PY_type_filename_template
    ``py{file_scope}type.{PY_impl_filename_suffix}``

PY_type_impl_template
    Names of functions for type methods such as ``tp_init``.
    ``{PY_prefix}{cxx_class}_{PY_type_method}{function_suffix}{template_suffix}``

PY_use_numpy
    Allow NumPy arrays to be used in the module.
    For example, when assigning to a struct-as-class member.

YAML_type_filename_template
    Default value for global field YAML_type_filename
    ``{library_lower}_types.yaml``


Format Fields
-------------

Each scope (library, class, function) has its own format dictionary.
If a value is not found in the dictionary, then the parent
scopes will be recursively searched.

Library
^^^^^^^

C_array_type
    Name of structure used to store information about an array
    such as its address and size.
    Defaults to *{C_prefix}SHROUD_array*.

C_bufferify_suffix
  Suffix appended to generated routine which pass strings as buffers
  with explicit lengths.
  Defaults to *_bufferify*

C_capsule_data_type
    Name of struct used to share memory information with Fortran.
    Defaults to *SHROUD_capsule_data*.

C_header_filename
    Name of generated header file for the library.
    Defaulted from expansion of option *C_header_filename_library_template*.

C_header_filename_suffix
   Suffix added to C header files.
   Defaults to ``h``.
   Other useful values might be ``hh`` or ``hxx``.

C_header_utility
   A header file with shared Shroud internal typedefs for the library.
   Default is ``types{library}.{C_header_filename_suffix}``.

C_impl_filename
    Name of generated C++ implementation file for the library.
    Defaulted from expansion of option *C_impl_filename_library_template*.

C_impl_filename_suffix:
   Suffix added to C implementation files.
   Defaults to ``cpp``.
   Other useful values might be ``cc`` or ``cxx``.

C_impl_utility
   A implementation file with shared Shroud helper functions.
   Typically routines which are implemented in C but called from
   Fortran via ``BIND(C)``.  The must have global scope.
   Default is ``util{library}.{C_header_filename_suffix}``.

C_local
    Prefix for C compatible local variable.
    Defaults to *SHC_*.

C_memory_dtor_function
    Name of function used to delete memory allocated by C or C++.

C_name_scope
    Underscore delimited name of namespace, class, enumeration.
    Used with creating names in C.
    Does not include toplevel *namespace*.

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

F_array_type
    Name of derived type used to store information about an array
    such as its address and size.
    Default value from option *F_array_type_template* which 
    defaults to *{C_prefix}SHROUD_array*.

F_C_prefix
    Prefix added to name of generated Fortran interface for C routines.
    Defaults to **c_**.

F_capsule_data_type
    Name of derived type used to share memory information with C or C++.
    Member of *F_array_type*.
    Default value from option *F_capsule_data_type_template* which 
    defaults to *{C_prefix}SHROUD_capsule_data*.

    Each class has a similar derived type, but with a different name
    to enforce type safety.

F_capsule_delete_function
    Name of type-bound function of *F_capsule_type* which will
    delete the memory in the capsule.
    Defaults to *SHROUD_capsule_delete*.

F_capsule_final_function
    Name of function used was ``FINAL`` of *F_capsule_type*.
    The function is used to release memory allocated by C or C++.
    Defaults to *SHROUD_capsule_final*.

F_capsule_type
    Name of derived type used to release memory allocated by C or C++.
    Default value from option *F_capsule_type_template* which 
    defaults to *{C_prefix}SHROUD_capsule*.
    Contains a *F_capsule_data_type*.

F_derived_member
    A *F_capsule_data_type* use to reference C++ memory.
    Defaults to *cxxmem*.

F_derived_member_base
    The *F_derived_member* for the base class of a class.
    Only single inheritance is support via the ``EXTENDS`` keyword in Fortran.

F_filename_suffix
    Suffix added to Fortran files.
    Defaults to ``f``.
    Other useful values might be ``F`` or ``f90``.

F_module_name
    Name of module for Fortran interface for the library.
    Defaulted from expansion of option *F_module_name_library_template*
    which is **{library_lower}_mod**.
    Then converted to lower case.

F_name_scope
    Underscore delimited name of namespace, class, enumeration.
    Used with creating names in Fortran.
    Does not include toplevel *namespace*.

F_impl_filename
    Name of generated Fortran implementation file for the library.
    Defaulted from expansion of option *F_impl_filename_library_template*.

F_pointer
    The name of Fortran wrapper local variable to save result of a 
    function which returns a pointer.
    The pointer is then set in ``F_result`` using ``c_f_pointer``.
    It must not be the same as any of the routines arguments.
    It defaults to *SHT_ptr*
    It is defined for each argument in case it is used by the
    fc_statements. Set to *SHPTR_arg_name*, where *arg_name* is the
    argument name.

F_result
    The name of the Fortran wrapper's result variable.
    It must not be the same as any of the routines arguments.
    It defaults to *SHT_rv*  (Shroud temporary return value).

F_result_ptr
    The name of a variable in the Fortran wrapper which holds the
    result of the C wrapper for functions which return a class instance.
    It will be type ``type(C_PTR)``.

..  XXX -  useful in wrappers to check for NULL pointers which may indicate error

F_result_capsule
    The name of the additional argument in the interface for functions
    which return a class instance.
    It will be type *F_capsule_data_type*.

F_string_result_as_arg
    The name of the output argument.
    Function which return a ``char *`` will instead be converted to a
    subroutine which require an additional argument for the result.
    See also *C_string_result_as_arg*.

F_this
   Name of the Fortran argument which is the derived type
   which represents a C++ class.
   It must not be the same as any of the routines arguments.
   Defaults to ``obj``.

file_scope
   Used in filename creation to identify library, namespace, class.

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
    The current C++ namespace delimited with ``::`` and a trailing ``::``.
    Used when referencing identifiers: ``{namespace_scope}id``.

nullptr
    Set to `NULL` or `nullptr` based on option *CXX_standard*.
    Always `NULL` when *language* is C.

PY_ARRAY_UNIQUE_SYMBOL
   C preprocessor define used by NumPy to allow NumPy to be
   imported by several source files.
    
PY_header_filename_suffix
   Suffix added to Python header files.
   Defaults to ``h``.
   Other useful values might be ``hh`` or ``hxx``.

PY_impl_filename_suffix
   Suffix added to Python implementation files.
   Defaults to ``cpp``.
   Other useful values might be ``cc`` or ``cxx``.

PY_module_init
    Name of module and submodule initialization routine.
    library and namespaces delimited by ``_``.
    Setting *PY_module_name* will update *PY_module_init*.

PY_module_name
    Name of generated Python module.
    Defaults to library name or namespace name.

PY_module_scope
    Name of module and submodule initialization routine.
    library and namespaces delimited by ``.``.
    Setting *PY_module_name* will update *PY_module_scope*.

PY_name_impl
    Name of Python wrapper implemenation function.
    Each class and namespace is implemented in its own function with file
    static functions.  There is no need to include the class or namespace in
    this name.
    Defaults to *{PY_prefix}{function_name}{function_suffix}*.

PY_prefix
    Prefix added to Python wrapper functions.

PY_result
    The name of the Python wrapper's result variable.
    It defaults to *SHTPy_rv*  (return value).
    If the function returns multiple values (due to *intent(out)*)
    and the function result is already an object (for example, a NumPy array)
    then **PY_result** will be **SHResult**.

file_scope
    library plus any namespaces.
    The namespaces listed in the top level variable *namespace* is not included in the value.
    It is assumed that *library* will be used to generate unique names.
    Used in creating a filename.

stdlib
    Name of C++ standard library prefix.
    blank when *language=c*.
    ``std::`` when *language=c++*.

YAML_type_filename
    Output filename for type maps for classes.

Enumeration
^^^^^^^^^^^

cxx_value
    Value of enum from YAML file.

enum_lower

enum_name

enum_upper

enum_member_lower

enum_member_name

enum_member_upper

flat_name
    Scoped name of enumeration mapped to a legal C identifier.
    Scope operator `::` replaced with `_`.
    Used with *C_enum_template*.

C_enum_member
    C name for enum member.
    Computed from *C_enum_member_template*.

C_value
    Evalued value of enumeration.
    If the enum does not have an explict value, it will not be present.

C_scope_name
    Set to *flat_name* with a trailing undersore.
    Except for non-scoped enumerations in which case it is blank.
    Used with *C_enum_member_template*.
    Does not include the enum name in member names for non-scoped enumerations.

F_scope_name
   Value of *C_scope_name* converted to lower case.
   Used with *F_enum_member_template*.

F_enum_member
    Fortran name for enum member.
    Computed from *F_enum_member_template*.

F_value
    Evalued value of enumeration.
    If the enum does not have an explict value, it is the previous value plus one.

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
   Defaults to the value *cxx_class* (usually the C++ class name) converted
   to lowercase.

F_name_assign
    Name of method that controls assignment of shadow types.
    Used to help with reference counting.

F_name_associated
    Name of method to report if shadow type is associated.
    If the name is blank, no function is generated.

F_name_final
    Name of function used in ``FINAL`` for a class.

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
    Used in generating names for C and Fortran and filenames.
    When the class is templated, it willl be converted to a legal identifier
    by adding the *template_suffix* or a sequence number.

    When *cxx_class* is set in the YAML file for a class, its value will be
    used in *class_scope*, *C_name_scope*, *F_name_scope* and *F_derived_name*.

cxx_type
    The namespace qualified name of the C++ class, including information
    from *template_arguments*, ex. ``std::vector<int>``.
    Same as *cxx_class* if *template_arguments* is not defined.
    Used in generating C++ code.

class_scope
    Used to to access class static functions.
    Blank when not in a class.
    ``{cxx_class}::``

C_prefix
    Prefix for C wrapper functions.
    The prefix helps to ensure unique global names.
    Defaults to the first three letters of *library_upper*.

PY_helper_prefix
    Prefix added to helper functions for the Python wrapper.
    This allows the helper functions to have names which will not conflict
    with any wrapped routines.
    When option *PY_write_helper_in_util* is *True*, *C_prefix* will
    be prefixed to the value to ensure the helper functions will not
    conflict with any routines in other wrapped libraries.

PY_type_obj
    Name variable which points to C or C++ memory.
    Defaults to *obj*.

PY_type_dtor
    Pointer to information used to release memory.

PY_PyTypeObject
    Name of `PyTypeObject` variable for a C++ class.
    Computed from option *PY_PyTypeObject*.

PY_PyTypeObject_base
    The name of `PyTypeObject` variable for base class of C++ class.
    Only single inheritance is support via the tp_base field of `PyTypeObject` struct.
    
Function
^^^^^^^^

C_call_list
    Comma delimited list of function arguments.

.. uses tabs

C_name
    Name of the C wrapper function.
    Defaults to evaluation of option *C_name_template*.

C_prototype
    C prototype for the function.
    This will include any arguments required by annotations or options,
    such as length or **F_string_result_as_arg**.  

.. uses tabs

C_return_type
    Return type of the C wrapper function.
    If the **return_this** field is true, then set to ``void``.
    
    Set to function's return type.

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

.. uses tabs

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
    String append to a generated function name.
    Useful to distinguish overloaded function and functions with default arguments.
    Defaults to a sequence number with a leading underscore
    (e.g. `_0`, `_1`, ...) but can be set
    by using the function field *function_suffix*.
    Multiple suffixes may be applied -- overloaded with default arguments.

LUA_name
    Name of function as known by LUA.
    Defaults to evaluation of option *LUA_name_template*.

template_suffix
   String which is append to the end of a generated function names
   to distinguish template instatiations.
   Default values generated by Shroud will include a leading underscore.
   i.e ``_int`` or ``_0``.

underscore_name
    *function_name* converted from CamelCase to snake_case.

Argument
^^^^^^^^

c_const
    ``const`` if argument has the *const* attribute.

c_deref
    Used to dereference *c_var*.
    ``*`` if it is a pointer, else blank.

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

cxx_addr
    Syntax to take address of argument.
    ``&`` or blank.

cxx_nonconst_ptr
    A non-const pointer to *cxx_addr* using `const_cast` in C++ or
    a cast for C.

cxx_member
    Syntax to access members of *cxx_var*.
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

size_var
    Name of variable which holds the size of an array in the
    Python wrapper.

Result
^^^^^^

cxx_rv_decl
    Declaration of variable to hold return value for function.


Variable
^^^^^^^^

PY_struct_context
   Prefix used to to access struct/class variables.
   Includes trailing syntax to access member in a struct
   i.e. ``.`` or ``->``.
   ``self->obj->``.
    

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

forward
    Forward declaration.
    Defaults to *None*.

typedef
    Initialize from existing type
    Defaults to *None*.

f_return_code
    Fortran code used to call function and assign the return value.
    Defaults to *None*.

f_to_c
    Expression to convert Fortran type to C type.
    If this field is set, it will be used before f_cast.
    Defaults to *None*.



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

