.. Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and
   other Shroud Project Developers.
   See the top-level COPYRIGHT file for details.

   SPDX-License-Identifier: (BSD-3-Clause)

Releases
========

Notes to help migrate between releases.

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
  
