.. Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and
   other Shroud Project Developers.
   See the top-level COPYRIGHT file for details.

   SPDX-License-Identifier: (BSD-3-Clause)

Releases
========

Notes to help migrate between releases.

Unreleased
----------

Changes to YAML input
^^^^^^^^^^^^^^^^^^^^^

* Added attribute *+funptr*. Uses ``type(C_FUNPTR)`` for
  function pointer arguments.

* Create an abstract interface for typedef statements which
  are function pointers. Previously, only function pointers
  arguments were supported.

* Attribute *+allocatable* is now *+deref(allocatable)*.
  This avoids setting *+allocatable* inconsistent with *+deref*.

* C wrappers can now be generated independent of Fortran wrappers
  instead of just as a side effect of creating Fortran Wrappers.

  As part of this effort some uses of *fstatements* in the YAML file
  must be changed.  The C wrapper created for the Fortran wrapper to call
  is now considered part of the Fortran wrapper processing.
  The *c_buf* label used with *fstatements* is now *f*.

* Rename some fields in Statements to allow C and Fortran entries to exist
  in the same group by consistently using a ``c_``, ``i_`` or ``f_`` prefix.
  This allows a single group to contains all the fields used for more complex
  conversions making it easier to follow the flow.

  This will change the name of fields in *fstatements* in an input YAML file.
  These are used to changed the default behavior of a wrapper.

c statements

=============   =============
Old Name        New Name
=============   =============
arg_call        c_arg_call
pre_call        c_pre_call
call            c_call
post_call       c_post_call
final           c_final
ret             c_return
temps           c_temps
local           c_local
f_arg_decl      i_arg_decl
f_result_decl   i_result_decl
f_result_var    i_result_var
f_module        i_module
f_import        i_import
=============   =============

f statements

=============   =============
Old Name        New Name
=============   =============
need_wrapper    f_need_wrapper
arg_name        f_arg_name
arg_decl        f_arg_decl
arg_c_call      f_arg_call
declare         f_declare
pre_call        f_pre_call
call            f_call
post_call       f_post_call
result          f_result
temps           f_temps
local           f_local
=============   =============

.. from vectors.yaml

.. code-block:: yaml

    fstatements:
      c:
        return_type: long
        ret:
        - return SHT_arg_cdesc->size;
      c_buf:
        return_type: long
        ret:
        - return SHT_arg_cdesc->size;
      f:
        result: num
        f_module:
          iso_c_binding: ["C_LONG"]
        declare:
        -  "integer(C_LONG) :: {F_result}"
        call:
        -  "{F_result} = {F_C_call}({F_arg_c_call})"              

is now:

.. code-block:: yaml

    fstatements:
      c:
        c_return_type: long
        c_return:
        - return SHT_arg_cdesc->size;
      f:
        c_return_type: long
        c_return:
        - return SHT_arg_cdesc->size;
        f_result: num
        f_module:
          iso_c_binding: ["C_LONG"]
        f_arg_decl:
        -  "integer(C_LONG) :: {F_result}"
        f_call:
        -  "{F_result} = {F_C_call}({F_arg_c_call})"              


* Added format field *f_c_suffix*. Used in format fields
  *C_name_template* and *F_C_name_template* to allow Fortran wrapper
  *to call a C function with additional mangling such as
  *C_cfi_suffix* and *C_bufferify_suffix*.  Previously this was
  *appended directly to format field *function_suffix*. If
  *C_name_template* or F_C_name_template* are explicitly set in the
  *YAML file then *f_c_suffix* should be included in the value.

.. See names.yaml

* Renamed format fields *hnamefunc*. These fields were added from the
  statement fields **c_helper** and **f_helper**, each a blank
  delimited list of names. A format field was added for each name with
  a 0-based suffix corresponding to the position in the list.
  Now, the format fields have the prefix of *c_helper_* or *f_helper_*
  followed by the helpers name. For example, *f_helper_copy_array*.
  This makes it easier to match the corresponding helper and will help
  when using statement mixin groups since the order of names will no
  longer matter.

* Changed statement fields *c_helper* and *f_helper* from a blank
  delimited list, into a YAML list.  If they are used in a
  *f_statements* section of a YAML file, they will need to be changed.
  This makes them more consistent with *f_temps* and *c_temps* which
  are also list of names.

  For example, from ``generic.yaml``

.. code-block:: yaml

    -      c_helper: ShroudTypeDefines
    +      c_helper:
    +      - ShroudTypeDefines

.. And easier to use in a mixin group by appending lists.

* Renamed some helpers to have more consistent names.
  Now the helpers and the function it defines may have different names.
  Use snake case for all helpers names (before about half used camel case).
  Continue to use camel case for function names.
  Remove *Shroud* from the helper name since that's redundant.
  Rename some functions from ``Str`` to ``Char`` to make clear when
  it's dealing with C++ types ``char`` vs ``std::string``.

.. Use the helper name in statements to make it easier to rename
   functions without renaming helpers.

.. list-table:: f statements
   :widths: 25 25
   :header-rows: 1

   * - Old Name
     - New Name
   * - ShroudStrAlloc
     - char_alloc
   * - ShroudStrArrayAlloc
     - char_array_alloc
   * - ShroudStrArrayFree
     - char_array_free
   * - ShroudStrBlankFill
     - char_blank_fill
   * - ShroudStrCopy
     - char_copy
   * - ShroudStrFree
     - char_free
   * - ShroudStrToArray
     - string_to_cdesc
   * - ShroudTypeDefines
     - type_defines


.. Structs in the C++ wrappers now accessed via  a ``using`` statement.
   The C structs which are created are only used by users of the header,
   not the implementation.
   As a side effect of this, the forward.yaml test no longer needs to define
   the *c_type* field since the C++ type will be used.

* Renamed some format fields to allow more control of argument names
  in wrappers.  The C wrapper continues to use *c_var* and *cxx_var*.
  The Fortran wrapper continues to use *f_var*, but if a different
  argument is needed to be passed to the C wrapper it is now *fc_var*
  instead of *c_var*.  The interface uses *i_var* instead of reusing
  *c_var*. Remove format field *F_C_var* since it is redundant with
  *i_var*.

.. The fmtc and fmtf dictionaries were merged and needed unique names
   instead of overloading c_var.

.. As part of creating better C specific wrappers (not intented to be
   called by Fortran, but need a modified API. For example, returning
   vectors), the fstatements field of a function in the YAML file has
   changed.  `c_buf` and `f` fields need to be merged.  A fstatements
   now has both the C and Fortran variables.

   Likewise, *patterns* used by *C_error_pattern* and local splicers
   use *buf* and *cfi* and will need to change.

* The *deref* attribute is no longer applied to the C wrapper.  When
  the function result had *+deref(scalar)* on a pointer result, a
  scalar was returned. The C wrapper will now return a pointer giving
  it the same prototype as the C++ library function.

.. The C wrapper used by the Fortran wrapper will return a scalar to
   avoid having to dereference it in the Fortran wrapper via
   c_f_pointer. And in the simpliest case, eliminates the need
   for the Fortran wrapper entirely.
  
New Features
^^^^^^^^^^^^

* Added *fmtdict* field to Fortran and C statement groups. Similar to
  *fmtdict* already in the Python statement groups. It allows format
  fields to be set explicitly in the statement group to override the
  any defaults.

* Support recursive structs. Allows trees to be build in structs.
* Add getter/setter for ``struct`` pointer fields in a struct.

.. Setting *deref* attribute on struct members will be used with the getter.
   Before only dimension was used.

Fixed
^^^^^

* Fixed the case of mixing default arguments with *fortran_generic*.
  The *fortran_generic* was restore arguments in the Fortran wrapper
  which were being trimmed by default arguments.

v0.13.0
-------

Changes
^^^^^^^

* Some generated wrapper names have been changed to be more consistent.
  Added format field *F_name_api*. It is controlled by option
  **F_API_case** which may be set to *lower*, *upper*, *underscore* or
  *preserve*.  Uses of format field *underscore_name* should be
  changed to *F_name_api*.  It's often used in name options such as
  **F_name_impl_template** and **F_name_generic_template**.

  Likewise, C API names are controlled by option **C_name_api**.  The
  default is *preserve*.  The previous behavior can be restored by
  setting option **C_API_case** to *underscore*.

  **F_API_case** defaults to *underscore* since Fortran is case insensitive.
  **F_C_case** defaults to *preserve* to make the C API closer to the C++ API.

* Changed the name of C and Python function splicer to use *function_name* instead
  of *underscore_name* to correspond to C++ library names.

* The *C_memory_dtor_function* is now written to the utility file,
  *C_impl_utility*.  This function contains code to delete memory from
  shadow classes. Previously it was written to file *C_impl_filename*.
  In addition, some helper functions are also written into this file.
  This may require changes to Makefiles to ensure this file is compiled.

* A single capsule derived type is created in the Fortran wrapper
  instead of one per class.  This is considered an implementation
  detail and a user of the wrapper will not access them directly.
  However, it may show up in splicer code.  It is used to pass values
  from the Fortran wrapper to the C wrapper.  The old type names may
  of been referenced in explicit splicer code.  In that case the name
  will need to be changed.  The format field
  *F_capsule_data_type_class* is replaced by *F_capsule_data_type*.
  The C wrapper continues to create a capsule struct for each class
  as a form of type safety in the C API.

* Class instance arguments which are passed by value will now pass the
  shadow type by reference. This allows the addr and idtor fields to be
  changed if necessary by the C wrapper.

* Replaced the *additional_interfaces* splicer with *additional_declarations*.
  This new splicer is outside of an interface block and can be used to add
  add a generic interface that could not be added to *additional_interfaces*.
  Existing *additional_interfaces* splicers can be converted to
  *additional_declarations* by wrapping the splicer with
  ``INTERFACE``/``END INTERFACE``.
  

New Features
^^^^^^^^^^^^

* Added support for C++ class inheritance.
  See :ref:`struct_class_inheritance`  

* Added the ability to treat a struct as a class.
  See :ref:`struct_object_oriented_c`

* Added the ability to declare members of a struct on
  individual ``decl`` lines in the YAML file similar to how
  class members are defined. Before the struct was defined
  in a single ``decl:``.

* Allow structs to be templated.

* Added the ability to declare variables using the ``enum`` keyword.
  C++ creates a type for each enumeration.

* Generate generic interface which allows a scalar or array to be
  passed for an argument.

* Process assumed-rank dimension attribute, *dimension(..)*.
  Create a generic interface using scalar and each rank.

* Added some support for Futher Interoperability with C.
  Used when option *F_CFI* is True (C/Fortran Interoperability).

* Support *deref(pointer)* for ``char *`` and ``std::string`` functions.
  Requires at least gfortran 6.1.0

* Added option F_trim_char_in. Controls where ``CHARACTER`` arguments
  are NULL terminated. If *True* then terminated in Fortran else in C.

* Added attribute *+blanknull* to convert a blank Fortran string into
  a NULL pointer instead of a 1-d buffer with ``'/0'``.
  Used with ``const char *`` arguments.
  This can be defaulted to True with the *F_blanknull* option.

* Added ``file_code`` dictionary to input YAML file. It contains
  directives to add header file and ``USE`` statements into generated files.
  These are collated with headers and ``USE`` statements added by typemaps,
  statements and helpers to avoid duplication.

* Allow typemaps with *base* as *integer* and *real* to be added to the
  input YAML file. This allows kind parameters to be defined via splicers
  then used by a typemap.  i.e. ``integer(INDEXTYPE)``

* Added option *C_shadow_result*. If true, the C wrapper will return a pointer
  to the capsule holding the function result. The capsule is also passed
  as an argument.  If false the function is ``void``.

* The getter for a class member function will return a Fortran pointer if
  the *dimension* attribute is added to the declaration.
  Likewise, the setter will expect an array of the same rank as *dimension*.
  Getter and setters will also be generated for struct fields which are pointers
  to native types. Option *F_struct_getter_setter* can be used to control their
  creation.

* Added ability to add *splicer* to ``typedef`` declarations.
  For example, to use the C preprocessor to set the type of the typedef.
  See typedefs.yaml for an example.

* Added support for out arguments which return a reference to a ``std::vector``
  or pointer to an array of ``std::string``.

* Create C and Fortran wrappers for typedef statements.
  Before ``typedef`` was treated as an alias.  ``typedef int TypeID`` would
  substitute ``integer(C_INT)`` for every use of ``TypeID`` in the Fortran wrapper.
  Now a parameter is created: ``integer, parameter :: type_id = C_INT``.
  Used as: ``integer(type_id) :: arg``.
  
Fixed
^^^^^

* Order of header files in *cxx_header* is preserved in the generated code.

* Create a generic interface even if only one *decl* is in the *fortran_generic* list.

* *generic_function* now creates a C wrapper for each Fortran wrapper.
  This causes each Fortran interface to bind to a different C function which
  fixes a compile error with xlf.

* Add generic interfaces for class methods.  Generic functions where only being added
  to the type-bound procedures.  ``class_generic(obj)`` now works instead of only
  ``obj%generic()``.

* Add continuations on Fortran ``IMPORT`` statements.

* Support an array of pointers - ``void **addr+rank(1)``.

*  Fix Fortran wrapper for ``intent(INOUT)`` for ``void **``.

* Promote wrap options (ex wrap_fortran) up to container when True
  (library, class, namespace). This allows wrap_fortran to be False at
  the global level and set True on a function and get a wrapper.
  Before a False at the global level would never attempt to do any
  wrapping.

* Better support for ``std::vector`` with pointer template arguments.
  For examples, ``<const double *>``.

* Parse ``class``, ``struct`` and ``enum`` as part of declaration.
  This allows ``typedef struct tag name`` to be parsed properly.
  
* Create type table earlier in parse. This allows recursive structs such as
  ``struct point { struct point *next; }`` to be parsed.
  
* Fixed issues in converting function names from CamelCase

  * Remove redundant underscore
    ``Create_Cstruct_as_class`` was ``c_create__cstruct_as_class`` now ``c_create_cstruct_as_class``
  * Add missing underscore
    ``AFunction`` was ``afunction`` now ``a_function``.
  
