.. Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and
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

write-statements BASE
       Write a file which contain the statements tree.
       Used for debugging.

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
  The order will be preserved when generating wrapper files.

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
   Controls mangling of C++ library names to C names
   via the format field *C_name_api*.
   Possible values are *lower*, *upper*, *underscore*, or *preserve*.
   Defaults to *preserve* and will be combined with *C_prefix*.
   For example, **C_name_template** includes ``{C_prefix}{C_name_scope}{C_name_api}``.

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

C_shadow_result
  If *true*, the api for the function result will be set to *capptr*,
  otherwise it will be set to *capsule*.  In both cases, the result is
  passed from Fortran to the C api as an additional argument. With
  *C_shadow_result* true, a pointer to the capsule is returned as the
  function result.  If *false*, the C wrapper is a ``void`` function.
  *capptr* acts more like C library functions such as ``strcpy`` which
  return a pointer to the result. *capsule* makes for a simpler
  Fortran wrapper implementation since the function result is not used
  since it is identical to the result argument.

class_baseclass
  Used to define a baseclass for a struct for *wrap_struct_as=class*".
  The baseclase must already be defined earlier in the YAML file.
  It must be in the same namespace as the struct.

.. example from struct.yaml
  
.. code-block:: yaml

    - decl: struct Cstruct_as_class
      options:
        wrap_struct_as: class
    - decl: struct Cstruct_as_subclass
      options:
        wrap_struct_as: class
        class_baseclass: Cstruct_as_class

 This is equivelent to the C++ code

 .. code-block:: c++

    class Cstruct_as_class;
    class Cstruct_as_subclass : public Cstruct_as_class;

The corresponding Fortran wrapper will have

.. code-block:: fortran

    type cstruct_as_class
      type(STR_SHROUD_capsule_data) :: cxxmem
    end type cstruct_as_class
    type, extends(cstruct_as_class) ::  cstruct_as_class
    end type cstruct_as_subclass

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

class_method
  Indicates that this function is a method for a struct.

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

F_API_case
   Controls mangling of C++ library names to Fortran names
   via the format field *F_name_api*.
   Possible values are *lower*, *upper*, *underscore*, or *preserve*.
   Defaults to *underscore* to convert ``CamelCase`` to ``camel_case``.
   Since Fortran is case insensitive, users are not required to
   respect the case of the C++ name.  Using *underscore* makes the
   names easier to read regardless of the case.

F_assumed_rank_min
  Minimum rank of argument with assumed-rank.
  Defaults to 0 (scalar).

F_assumed_rank_max
  Maximum rank of argument with assumed-rank.
  Defaults to 7.

F_blanknull
  Default value of attribute *+blanknull* for ``const char *``
  arguments.  This attribute will convert blank Fortran strings
  to a ``NULL`` pointer.

F_CFI
  Use the C Fortran Interface provided by *Futher Interoperability with C*
  from Fortran 2018 (initially defined in TS29113 2012).

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
  the return type since Fortran does not distinguish generics based on
  return type (similar to overloaded functions based on return type in
  C++).

.. XXX should also be set to false when the templated argument in
   cxx_template is part of the implementation and not the interface.

F_default_args
  Decide how to handle C++ default argument functions.
  See :ref:`DefaultArguments`.

  generic
      Create a wrapper for each variation from all arguments
      to no arguments defaulted.  In Fortran, create a generic
      interface.
  optional
      Make each default argument as a Fortran ``OPTIONAL`` argument.
  require
      Require all arguments to be provided to the wrapper.

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

F_return_fortran_pointer
  Use ``c_f_pointer`` in the Fortran wrapper to return 
  a Fortran pointer instead of a ``type(C_PTR)``
  in routines which return a pointer.
  It does not apply to ``char *``, ``void *``, and routines which return
  a pointer to a class instance.
  Defaults to *true*.

F_string_len_trim
  For each function with a ``std::string`` argument, create another C
  function which accepts a buffer and length.  The C wrapper will call
  the ``std::string`` constructor, instead of the Fortran wrapper
  creating a ``NULL`` terminated string using ``trim``.  This avoids
  copying the string in the Fortran wrapper.
  Defaults to *true*.

F_struct_getter_setter
  If true, a getter and setter will be created for struct members
  which are a pointer to native type. This allows a Fortran pointer
  to be used with the field instead of having to deal with the
  ``type(C_PTR)`` directly.
  Default to *true*

F_trim_char_in
  Controls code generation for ``const char *`` arguments.
  If *True*, Fortran perform a ``TRIM`` and concatenates
  ``C_NULL_CHAR``.  If *False*, it will be done in C.  If the only
  need for the C wrapper is to null-terminate a string (wrapping a c
  library and no other argument requires a wrapper), then the C
  wrapper can be avoid by moving the null-termination action to
  Fortran.
  Default is *True*.

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

LUA_API_case
  Controls mangling of C++ library names to Lua names
  via the format field *LUA_name_api*.
  Possible values are *lower*, *upper*, *underscore*, or *preserve*.
  Defaults to *preserve*.

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
dictionary to create names in the generated code.

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
    ``{C_prefix}{C_name_scope}{C_name_api}{function_suffix}{f_c_suffix}{template_suffix}``

C_name_typedef_template
    ``{C_prefix}{C_name_scope}{typedef_name}``
    
F_C_name_template
    ``{F_C_prefix}{F_name_scope}{F_name_api}{function_suffix}{f_c_suffix}{template_suffix}``

F_abstract_interface_argument_template
   The name of arguments for an abstract interface used with function pointers.
   Defaults to ``{F_name_api}_{argname}``
   where *argname* is the name of the function argument.
   see :ref:`DeclAnchor_Function_Pointers`.

F_abstract_interface_subprogram_template
   The name of the abstract interface subprogram which represents a
   function pointer.
   Defaults to ``arg{index}`` where *index* is the 0-based argument index.
   See :ref:`DeclAnchor_Function_Pointers`.

F_array_type_template
   ``{C_prefix}SHROUD_array``
   
F_capsule_data_type_template
    Name of the derived type which is the ``BIND(C)`` equivalent of the
    struct used to implement a shadow class (**C_capsule_data_type**).
    All classes use the same derived type.
    Defaults to ``{C_prefix}SHROUD_capsule_data``.

F_capsule_type_template
    ``{C_prefix}SHROUD_capsule``

F_derived_name_template
    Defaults to ``{F_name_api}``.
    
F_enum_member_template
    Name of enumeration member in Fortran wrapper.
    ``{F_name_scope}{enum_member_lower}``
    Note that *F_enum_template* does not exist since only the members are 
    in the Fortran code, not the enum name itself.

F_name_generic_template
    ``{F_name_api}``

F_impl_filename_library_template
    ``wrapf{library_lower}.{F_filename_suffix}``

F_name_impl_template
    ``{F_name_scope}{F_name_api}{function_suffix}{template_suffix}``

F_module_name_library_template
    ``{library_lower}_mod``

F_module_name_namespace_template
    ``{file_scope}_mod``

F_name_function_template
    ``{F_name_api}{function_suffix}{template_suffix}``

F_typedef_name_template
    ``{F_name_scope}{F_name_api}``
    
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

SH_class_getter_template
    Name of generated getter function for class members.
    The wrapped name will be mangled futher to distinguish scope.
    Defaults to ``get_{wrapped_name}``.

SH_class_setter_template
    Name of generated setter function for class members.
    The wrapped name will be mangled futher to distinguish scope.
    Defaults to ``set_{wrapped_name}``.

SH_struct_getter_template
    Name of generated getter function for struct members.
    The wrapped name will be mangled futher to distinguish scope.
    Defaults to ``{struct_name}_get_{wrapped_name}``.

SH_struct_setter_template
    Name of generated setter function for struct members.
    The wrapped name will be mangled futher to distinguish scope.
    Defaults to ``{struct_name}_set_{wrapped_name}``.

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
    Name of structure used to store metadata about an array
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

C_name_api
    Root name that is used to create various names in the C API.
    Defaulted by the **C_API_case** option with values
    *lower*, *upper*, *underscore*, or *preserve*.
    If set explicitly then **C_API_case** will have no effect.

    May be blank for namespaces to avoid adding the name to
    *C_name_scope*.

C_name_scope
    Underscore delimited name of namespace, class, enumeration.
    Used to 'flatten' nested C++ names into global C identifiers.
    Ends with trailing underscore to allow the next scope to be appended.
    Does not include toplevel *namespace*.
    For example, **C_name_template** includes ``{C_prefix}{C_name_scope}{C_name_api}``.

    *C_name_scope* will replace *class_name* with the instantiated *class_name*.
    which will contain a template arguments.

    This is a computed using *C_name_api* and should not be set explicitly.

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
    Name of derived type used to store metadata about an array
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

F_name_api
    Root name that is used to create various names in the Fortran API.
    Controlled by the **F_API_case** option with values
    *lower*, *upper*, *underscore* or *preserve*.
    Used with options **templates F_C_name_template**, **F_name_impl_template**,
    **F_name_function_template**, **F_name_generic_template**,
    **F_abstract_interface_subprogram_template**, **F_derived_name_template**,
    **F_typedef_name_template**.

F_name_scope
    Underscore delimited name of namespace, class, enumeration.
    Used with creating names in Fortran.
    Ends with trailing underscore to allow the next scope to be appended.
    Does not include toplevel *namespace*.

    This is a computed using *F_name_api* and should not be set explicitly.
    
F_impl_filename
    Name of generated Fortran implementation file for the library.
    Defaulted from expansion of option *F_impl_filename_library_template*.

F_result
    The name of the Fortran wrapper's result variable.
    It must not be the same as any of the routines arguments.
    It defaults to *SHT_rv*  (Shroud temporary return value).

F_result_ptr
    The name of the variable used with api *capptr* for the
    function result for arguments which create a shadow type.
    Defaults to ``SHT_prv``, pointer to return value.
    Used by option *C_shadow_result*.

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

LUA_name_api
    Root name that is used to create various names in the Lua API.
    Defaulted by the **LUA_API_case** option with values
    *lower*, *upper*, *underscore*, or *preserve*.
    If set explicitly then **LUA_API_case** will have no effect.

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
    Computed from option *C_enum_member_template*.

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
    Computed from option *F_enum_member_template*.

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
    Computed from option *F_derived_name_template*.

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
    Arguments are tab delimited to aid in creating continuations.

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
    ``subroutine`` or ``function`` for the ``bind(C)`` interface.
    The C wrapper funtion may be different Fortran wrapper function since
    some function results may be converted into arguments.

F_C_var
    Name of dummy argument in the ``bind(C)`` interface.

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

f_c_suffix
    Set by Shroud to allow the Fortran wrapper to call a C wrapper
    with additional mangling.  Usually set to the value of
    *C_bufferify_suffix* or *C_cfi_suffix*.
    
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

c_array_shape

c_array_size

c_array_size2
  The *dimension* attributes multiplied together.

c_char_len
  The value of the *len* attribute.
  It willl be evalued in the C wrapper.
  Defaults to 0 to indicate no length given.

c_blanknull
   Used as argument to ``ShroudStrAlloc`` to determine if a
   blank string, trimmed length is 0, should be a NULL pointer
   instead of an empty C string -- ``'\0'``.
   Set via attribute *+blanknull* on a ``const char *`` argument.
   Should be ``0`` or ``1``.

c_const
    ``const`` if argument has the *const* attribute.

c_deref
    Used to dereference *c_var*.
    ``*`` if it is a pointer, else blank.

c_var
    The C name of the argument.

.. XXX these fields are creatd by the *temps* or *local* statements field.
    
.. c_var_len
    Function argument generated from the *len* annotation.
    Used with char/string arguments.
    Set from option **C_var_len_template**.

.. c_var_size
    Function argument generated from the *size* annotation.
    Used with array/std::vector arguments.
    Set from option **C_var_size_template**.

.. c_var_trim    c_local_trim
    Function argument generated from the *len_trim* annotation.
    Used with char/string arguments.
    Set from option **C_var_trim_template**.

c_var_cdesc
    Name of variable of type ....

c_var_cdesc2
    
c_var_extents

c_var_lower

chelper_*
    Helper name for a function.
    Each name in statements *c_helper* will create a format name
    which starts with *chelper_* and end with the helper name.
    It will contain the name of the C function for the helper.
    Used by statements *c_pre_call* and *c_post_call* statements.

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
    The template parameters for templated arguments.
    ``std::vector<{cxx_T}>``

cxx_type
    The C++ type of the argument.

cxx_var
    Name of the C++ variable.

size_var
    Name of variable which holds the size of an array in the
    Python wrapper.

fmtc
""""

Format strings used with C wrappers.
Set for each argument.

fmtf
""""

Format strings used with Fortran wrappers.
Set for each argument.

c_var
    The name of the argument passed to the C wrapper.
    This is initially the same as *f_var* but when the
    statement field *c_local_var* is true, another name
    will be generated of the form ``SH_{f_var}``.
    A declaration will also be added using typemap.f_c_type.

default_value
    The value of a C++ default value argument.

.. XXX - only defined for native types (integer, real)    

f_array_allocate
    Fortran shape expression used with ``ALLOCATE`` statement when
    *dimension* attribute is set.
    For example, attribute  *+dimension(10)* will create ``(10)``.

f_array_shape
   Shape of array for use with ``c_f_pointer``.
   For example, attribute *+dimension(10)* will create``,\t SHT_rv_cdesc%shape(1:1)``.
   The leading comma is used since scalar will not add a ``SHAPE`` argument to ``c_f_pointer``.

f_assumed_shape
   Set when *rank* attribute is set to the corresponding shape.
   ``rank=1`` sets to ``(:)``,
   ``rank=2`` sets to ``(:,:)``, etc.
   May also be set to ``(..)`` when attribute *+dimension(..)* is used
   and option *F_CFI* is True.

f_capsule_data_type
    The name of the derived type used to share memory information with C or C++.
    *F_capsule_data_type* for the argument type.

f_cdesc_shape
    Used to assign the rank of a Fortran variable to a cdesc variable.
    It will be blank for a scalar.
    ex: ``\nSHT_arg_cdesc%shape(1:1) = shape(arg)``

f_char_len
    Defaults to ``:`` for defered length used with allocatable variables.
    Used in statements as ``character({f_char_len)``.

f_char_type
    Character type used in ``ALLOCATE`` statements.
    Based on *len* attributes.
    Defaults to blank for ``CHARACTER`` types which have an explicit length
    in the type declaration - ``CHARACTER(20)``..
    Otherwise set to ``character(len={c_var_cdesc}%elem_len) :: `` which
    uses the length computed by the C wrapper and stored in elem_len.
    For example, find the maximum length of strings in a ``char **`` argument.
    Used in statements as ``allocate({f_char_type}(f_var})``.
    
f_declare_shape_prefix

f_declare_shape_array

f_derived_type
   Derived type name for shadow class.

f_get_shape_array

f_kind
    Value from typemap.  ex ``C_INT``.
    Can be used in *CStmts.f_module*.

f_pointer_shape

f_shape_var

f_type
    Value from typemap.  ex ``integer(C_INT)``.

f_type_module
    Module name for *f_type*.

f_var
    Fortran variable name for argument.

fhelper_*
    Helper name for a function.
    Each name in statements *f_helper* will create a format name
    which starts with *fhelper_* and end with the helper name.
    It will contain the name of the Fortran function for the helper.
    Used by statements *f_pre_call* and *f_post_call* statements.

i_dimension
    Dimension used in ``bind(C)`` interface.
    May be assumed-size, ``(*)`` or assumed-rank, ``(..)``.

i_type
    Used with Fortran interface.

size
    Expression to compute size of array argument using ``SIZE`` intrinsic.

fmtl
""""

Format strings used with Lua wrappers.

fmtpy
"""""

Format strings used with Python wrappers.

array_size
    Dimensions multipled together.
    ``dimension(2,3)`` creates ``(2)*(3)``.

rank
    Attribute value for *rank*.


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

