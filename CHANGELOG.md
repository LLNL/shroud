
# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.1.0/)
and this project adheres to [Semantic Versioning](http://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Changed
- Changed some entries in c_statements and f_statements to
  consistently prefix with ``f_``, ``i_`` or ``c_``. This allows one
  group to describe the entire Fortran wrapper, Fortran interface, C
  wrapper flow of a variable.
- Require an exact match when looking up statements for Fortran and C
  wrappers.  Before it used a trie tree to match 'close enough'
  statements.  This would sometimes result in different matches after
  adding new tree leafs which ended up changing the tree traversal.
- Added deref(arg) to deal with Fortran wrappers which return a
  function result via an argument. Used when
  fmt.F_string_result_as_arg is defined.  Eliminated
  meta["is_result"].
- Remove stmt.c_result_var. Remove custom code in wrapf and replaced
  with data in FStmts.
- *f_temps* and *f_local* now create format fields which start with **f**
  instead of **c**. They are intented to be used the Fortran wrapper
  and are based off of *f_var* name.

### Removed
- Removed attribute *+cdesc*. Replace by *+api(cdesc)*

## v0.13.0 - 2023-08-23
### Added
- Python hello-world-python example for a simple library in examples directory.
- Support for class inheritance.  Fortran uses the `EXTENDS` keyword.
  Python uses `PyTypeObject.tp_base` field.
- Ability to treat a struct as a class.
  Useful with C code which does not support ``class`` directly.
  Add options *wrap_class_as*, *wrap_struct_as*, *class_baseclass*, *class_ctor* and *class_method*.
  Attribute *+pass* defines the passed-object dummy argument. Defines ``PASS``
  Fortran keyword with type-bound procedures.
- ``struct`` can now be defined the same as a ``class`` by using a
  *declarations* field in the YAML file.  The entire struct can
  continue to be declared in a single *decl* field as before.  This
  makes it easier to define opaque structs where the members are not
  listed. Useful with *wrap_struct_as=class*.
- Added the ability to declare variables using the ``struct`` keyword.
  ``struct tname vname;``. In this case the semicolon is required
  to distinguish the *decl* entry from defining a structure.
- Added the ability to declare variables using the ``enum`` keyword.
  C++ creates a type for each enumeration.
- Support `base: struct` in the `typemap` field of the YAML file. This
  allows structs wrapped outside of the current YAML file to be used.
- Generate generic interface which allows a scalar or array to be
  passed for an argument.
- Allow structs to be templated.
- Process assumed-rank dimension attribute, *dimension(..)*.
  Create a generic interface using scalar and each rank.
- Added support for Futher Interoperability with C.
  Used when option *F_CFI* is True (C/Fortran Interoperability).
- Support *deref(pointer)* for ``char *`` and ``std::string`` functions.
  Requires at least gfortran 6.1.0
- Added option F_trim_char_in. Controls where ``CHARACTER`` arguments
  are NULL terminated. If *True* then terminated in Fortran else in C.
- Added attribute *+blanknull* to convert a blank Fortran string into
  a NULL pointer instead of a 1-d buffer with ``'/0'``.
  Used with ``const char *`` arguments.
  This can be defaulted to True with the *F_blanknull* option.
- Added ``file_code`` dictionary to input YAML file. It contains
  directives to add header file and ``USE`` statements into generated files.
  These are collated with headers and ``USE`` statements added by typemaps,
  statements and helpers to avoid duplication.
- Allow typemaps with *base* as *integer* and *real* to be added to the
  input YAML file. This allows kind parameters to be defined via splicers
  then used by a typemap.  i.e. ``integer(INDEXTYPE)``
- Added option *C_shadow_result*. If true, the C wrapper will return a pointer
  to the capsule holding the function result. The capsule is also passed
  as an argument.  If false the function is ``void``.
- The getter for a class member function will return a Fortran pointer if
  the *dimension* attribute is added to the declaration.
  Likewise, the setter will expect an array of the same rank as *dimension*.
  Getter and setters will also be generated for struct fields which are pointers
  to native types. Option *F_struct_getter_setter* can be used to control their
  creation.
- Added ability to add *splicer* to ``typedef`` declarations.
  For example, to use the C preprocessor to set the type of the typedef.
  See typedefs.yaml for an example.
- Added support for out arguments which return a reference to a ``std::vector``
  or pointer to an array of ``std::string``.

### Fixed
- yaml extensions supported include .yml in addition to the previous .yaml
- Order of header files in *cxx_header* is preserved in the generated code.
- Struct in an inner namespace using Py_struct_arg=numpy is now properly wrapped.
- Support an array of pointers - ``void **addr+rank(1)``.
- Fix Fortran wrapper for ``intent(INOUT)`` for ``void **``.
- Create generic interface even if only one *decl* in *fortran_generic* list.
- Promote wrap options (ex wrap_fortran) up to container when True
  (library, class, namespace). This allows wrap_fortran to be False at
  the global level and set True on a function and get a wrapper.
  Before a False at the global level would never attempt to do any
  wrapping.
- *generic_function* now creates a C wrapper for each Fortran wrapper.
  This causes each Fortran interface to bind to a different C function which
  fixes a compile error with xlf.
- Add continuations on Fortran ``IMPORT`` statements.
- Better support for ``std::vector`` with pointer template arguments.
  For examples, ``<const double *>``.
- Parse ``class``, ``struct`` and ``enum`` as part of declaration.
  This allows ``typedef struct tag name`` to be parsed properly.
- Create type table earlier in parse. This allows recursive structs such as
  ``struct point { struct point *next; }`` to be parsed.
- Fixed issues in converting function names from CamelCase
  * Remove redundant underscore
    ``Create_Cstruct_as_class`` was ``c_create__cstruct_as_class`` now ``c_create_cstruct_as_class``
  * Add missing underscore
    ``AFunction`` was ``afunction`` now ``a_function``.
- Add generic interfaces for class methods.  Generic functions where only being added
  to the type-bound procedures.  ``class_generic(obj)`` now works instead of only
  ``obj%generic()``.
- Replaced the *additional_interfaces* splicer with *additional_declarations*.
  This new splicer is outside of an interface block and can be used to add
  add a generic interface that could not be added to *additional_interfaces*.

### Changed
- Changed name of C and Python function splicers to use *function_name* instead of
  *underscore_name*.
- Changed default name mangling for C wrapper functions. Before it used the
  *underscore_name* format field which converted ``CamelCase`` to ``camel_case``.
  Added format field *C_name_api*, which is controlled by option *C_API_case*.
  The default is now to preserve the case of the C++ library routine.
  The previous behavior can be restored by setting option ``C_API_case: underscore``.
  Removed format fields *lower_name*, *upper_name* and *underscore_case*.
- Changed default name mangling for LUA wrappers. Before it used the
  *underscore_name* format field. Now controlled by option *LUA_API_case* which defaults
  to *preserve* and sets *fmt.LUA_name_api*.
- Changed default name mangling for Fortran derived type names. Before it used
  the *lower_name* format field which converted the ``CamelCase`` to ``camelcase``.
  Added format field *F_name_api*, which is controlled by option *F_API_case*.
  *F_name_api* is used by other options to define the Fortran name mangling consistently.
- The *C_memory_dtor_function* is now written to the utility file,
  *C_impl_utility*.  This function contains code to delete memory from
  shadow classes. Previously it was written file *C_impl_filename*.
  This may require changes to Makefiles.
- Class instance arguments which are passed by value will now pass the
  shadow type by reference. This allows the addr and idtor fields to be
  changed if necessary by the C wrapper.
- Create C and Fortran wrappers for typedef statements.
  Before ``typedef`` was treated as an alias.  ``typedef int TypeID`` would
  substitute ``integer(C_INT)`` for every use of ``TypeID`` in the Fortran wrapper.
  Now a parameter is created: ``integer, parameter :: type_id = C_INT``.
  Used as: ``integer(type_id) :: arg``.

### Removed
- Removed format field *F_capsule_data_type_class*.
  Create a single capsule derived type in Fortran instead of one per class.

## v0.12.2 - 2020-08-04
### Added
- Added std::string scalar arguments.

### Fixed
- Mangle names of Fortran helper functions with *C_prefix*.  This
  fixes a conflict when two Shroud generated wrappers are used in the
  same subprogram.  Ideally these helpers would be `PRIVATE`, but
  gfortran does not allow `PRIVATE` and `BIND(C)` together.
- Mangle names of Fortran derived types for capsules with *C_prefix* to avoid
  name conflicts.
- Correctly count Python arguments when there are default arguments.
  Do not include *intent(out)* arguments.

## v0.12.0 - 2020-06-29
### Added
- Option `C_force_wrapper` to create a C wrapper.
  This can be used when the function is actually a macro or function pointer.
- Added option CXX_standard.
  If *2011* or greater then `nullptr` is used instead of `NULL`.
  This makes clang-tidy happy.
- Create a setup.py for Python wrappers.
- Wrap pointer members in structs when PY_struct_arg="class".
- Use py_statements 'py_ctor' to create a struct-as-class constructor
  for Python which will assign to struct members.
- Add option *PY_write_helper_in_util* to write all helper functions
  into the file defined by format PY_utility_filename.  Useful when there
  are a lot of classes which may create lots of duplicate helpers.
- Parse array syntax for variables and struct members.
- Change Python setter and getter functions to be driven by py_statements.
- Parse `(void)` C prototype to indicate no parameters.

### Changed
- *intent(in)* pointer arguments now use the *rank* attribute instead of
  using the *dimension* attribute to defined Fortran assumed-shape.
  ex. ``dimension(:,:)`` should be ``rank(2)``.
  The dimension attribute must be a list of expressions and should not
  be assumed-shape or assumed-length and are used with *intent(out)* arguments.
- Pointer arguments default to ``intent(inout)`` instead of ``intent(in)``.
  ``const`` pointers remain ``intent(in)``.
- C++ class constructors create a generic interface in Fortran with the same name
  as the derived-type name.  Before it used the +name attribute in the generic name.
  This attribute is still used to create the actual function name.

### Fixed
- Inline splicers (defined as part of a decl) will be used before a
  splicer with the same name defined in a *splicer_code* or file splicer.
- C splicers were not looked up using the C wrapper name properly.
  They were using the C++ function name. This was a problem since a
  C++ function may produce several wrapper via overloading or
  adding bufferify arguments.
- Update fortran_generic generated functions to use the format statement
  in the fortran_generic clause. This allows the format values to be used
  in fstatement clauses.
````
fortran_generic:
- decl: (int *value)
  format:
    stype: int
````
- Allow multi-line scalar values to be used in YAML for inline splicers
  and fstatements. Convert empty lines from the YAML inserted None into
  an empty string in inline splicers and fstatements.
```
splicer:
  c: |
    ! lines of code
  c_buf:
  -  Next line is blank, not None
  -
```
- Allow wrap_c=False and wrap_fortran=True. Useful when the Fortran
  wrapper is defined by a splicer and the C wrapper is not needed.
- Improved support for templates in the Python wrappers.
- Added define for PyInt_FromSize_t for Python 3.
- Do not write Python utility file if it is empty.
- PY_struct_arg now applies to the struct. This allows two structs to use
  "class" and "numpy" in the same YAML file.
- Prevent duplicate helpers from being created. Helpers which are implemented
  in C but called from Fortran are written to a new file define by format
  *C_impl_utility*.
- Set idtor properly for constructors. Memory was not being released since
  idtor was 0.

## v0.11.0 - 2020-01-08
### Added
- A C++ namespace corresponds to a module in Fortran and Python.
  Options *flatten_namespace* and *F_flatten_namespace* can be used
  to reproduce the previous behavior.
- Added format dictionary field *template_suffix*.
- Added the **F_create_bufferify_function** option.
- Support `true` and `false` with implied attribute.
- Process 'void **' argument as 'type(C_PTR)' by reference.
- Added assumedtype attribute to use Fortran assumed-type declaration,
  `type(*)`, for `void *` arguments.
- Added external attribute to use with function pointers.
  Uses EXTERNAL statement instead of abstract interface to allow any function
  to be used as the dummy argument since the interface is not preserved.
- Added charlen attribute to use with 'char *arg+intent(out)' argument.
  Used to tell the Python wrapper the length of the char argument.
- Parse `enum class` and `enum struct`.
- Allow enum values to be an expression. Useful when an enum value is
  defined by previous enum values.
- Add option PY_array_arg to control how arrays are represented.
  Values can be *numpy* or *list*.
- Add option PY_struct_arg to control how structs are represented.
  Values can be *numpy* or *class*.
- Add command line option --options to set a top level option.
- Add command line option --language. May be c or c++.
  Replaces any language directive in the YAML file.
- Wrap functions which return a std::vector.

- Add *fstatements* to decl to add additional statement fields
  for the funtion result.
- Change Fortran splicer to include code from declare, pre_call, call,
  and post_call fields from cf_statements.
  Declarations for arguments and results are outside the splicer
  since they are controlled by the decl and attributes.

### Changed
- Default of library name from *default_library* to *library*.
- Renamed option *C_header_helper_template* to *C_header_utility_template*.
  Renamed option *PY_helper_filename_template* to *PY_utililty_filename_template*.
  This is to avoid confusion with helper functions which have file static scope.
- Changed C_name_template to use *C_name_scope* instead of *class_prefix*.
  *C_name_scope* is case sensitive while *class_prefix* was lower case.
  Option C_API_case can be set to *lower* to reproduce previous behavior.
  C_name_scope will also contain internal namespaces.
- Changed filename templates to use *C_file_scope* instead of *class_class*.
  This will include internal namespaces in the file name.
- Changed how the *fortran_generic* arguments are specified to allow multiple arguments
  and to associate attributes with arguments.
  previous format:
```
  - decl: void GenericReal(double arg)
    fortran_generic:
       arg:
       - float
       - double
```
  new format:
```
  - decl: void GenericReal(double arg)
    fortran_generic:
       - decl: (float arg)
         function_suffix: float
       - decl: (double arg)
         function_suffix: double
```
- Replace format fields C_code and F_code with splicers within the declaration.
  Consistent with other ways to define splicers.
  previous format:
```
   - decl: bool isNameValid(const std::string& name)
     format:
       C_code:  "return name != NULL;"
       F_code:  rv = name .ne. " "
```
  new format:
```
   - decl: bool isNameValid(const std::string& name)
     splicer:
        c:
        - "return name != NULL;"
        f:
        - 'rv = name .ne. " "'
```
- Replace format field C_finalize with statements clause final.
  previous_format:
```
  format:
    C_finalize_buf: delete {cxx_var};
```
  new_format:
```
  fstatements:
    c_buf:
      final:
      - delete {cxx_var};
```

### Fixed
- C++ function arguments which pass a class by value.
- C struct arguments passed to a bufferify function.
  It was using C++ casting.  Instead there is no C wrapper for a struct
  like there is with a Class since it is already compatible with Fortran.
- The Python wrapper for function which return bool and have other intent(out)
  arguments where not compiling.
- Changed enum option templates to include namespace as well as class name.
  Non-scoped enums only use *C_prefix* on enum member names.
- Passing a struct by value in Fortran wrapper for C library.
- Use IMPORT statement when creating an abstract interface with derived type
  argument which is defined by the library.

### Removed
- Format field *class_prefix*. Replace with *C_name_scope* or *F_name_scope*
  which includes the namespace.
- Option F_module_per_class. Now a namespace corresponds to a Fortran module.
- Format field *C_return_type* when set in the YAML file to change the
  return type. Use statement.return_type to modify the return type of
  a wrapper. It is still a format field with the actual return type.
- Format field *C_return_code*.  Replaced with statement.ret clause.
- Format fields *C_pre_call*, *C_call_code*, and *C_post_call*.
  They existed to help with splicers. statements now provide more control.
- Format field *C_post_call_pattern*.

## v0.10.1 - 2018-08-07
### Fixed
- Add a capsule derived type for each class to match the C struct.
  This fixes a compile error on xlf.

## v0.10.0 - 2018-08-01
### Added
- Added more support for unsigned types.
  Note that Fortran does not support unsigned types.
  ex. 'unsigned int' is mapped to C_INT.
- Add size based types: int8_t, int16_t, int32_t, int64_t.
- Add support for C++ structs.
  Fortran creates a derived type with ``bind(C)``.
  Python uses NumPy to unpack fields of struct.
- Wrap member variables in classes.
  Fortran and C create getter and setter functions.
  In Python, create a descriptor for each member.
- Allow C++ functions which return pointer to
  create a wrapper with the POINTER attribute.
  See option **return_scalar_pointer**.
- Support returning a class instance with C and Fortran wrapper.
  Allocate and release a wrapper controlled copy of instance.
- Added deref attribute to control how a pointer is returned to Fortran:
  allocatable, pointer, raw (type(C_PTR)).
- Support ALLOCATABLE CHARACTER function as the default for char* and std::string
  functions. Unless +len attribute or F_string_result_as_arg is used.
- Add owner attribute to define who should release memory: library or caller.
- Add free_pattern attribute to define custom code to release memory.
  For example, if the variable is reference counted.
- Support multiple template parameters.
- Support class templates.

### Changed
- Change C wrappers of shadow classes to return a pointer to a capsule_data_type
  instead of a `void *`.  This struct contains the pointer to the shadow class
  and information to deallocate it.  A pointer to the struct must also be
  passed in as the final argument to the wrapper.
- Change how function template arguments are specified to reflect C++ syntax.
  Also allow options and format to be added to an instantiation.
previous format:
```
  - decl: void Function7(ArgType arg)
    cxx_template:
      ArgType:
      - int
      - double
```
new format:
```
  - decl: |
        template<typename ArgType>
        void Function7(ArgType arg)
    cxx_template:
    - instantiation: <int>
    - instantiation: <double>
```

### Removed
- Remove +pure attribute.
  Also remove the feature where a string C++ function would be called twice,
  once to get length and once for values.  Instead use ALLOCATABLE string.

## v0.9.0 - 2018-04-04
### Added
- Support class static methods.
- Add enum as a declaration.
```
     - decl: enum color { RED, GREEN, BLUE }
```
- Add typedef as a declaration.
```
      - decl: typedef int TYP
```
### Changed
- base name 'wrapped' changed to 'shadow'.  This may appear in YAML file.
- Change generated code to prefix symbols with the namespace, `outer::function()`,
  instead of adding namespace statements, `namespace outer { }`.
  Types also require namespaces in `cxx_type` and `c_to_cxx` entries.
  The wrappers use the namespace but do not add to it.
- Change format of YAML file to generalize declarations and provide
  access to namespaces.
```
     # Old
     functions:
     - decl: void foo()
     # New
     declarations:
     - decl: void foo()

     # Old
     namespace: tutorial
     functions:
     - decl: void bar()
     # New
     declarations:
     - decl: namespace tutorial
       declarations:
       - decl: void bar()

     # Old
     types:
       CustomType:
          typedef: int
     # New
     declarations:
     - typedef: CustomType
       fields:
         typedef: int
```

## v0.8.0 - 2018-02-26
### Added
- Support for function pointer arguments.
- Improve support for Python wrappers, including NumPy support.
  Use constructor and destructor with extension types.
- Added implied attribute used to compute value of an argument which
  is in the C++ API but does not need to be explicit in the Fortran API.
- Added allocatable attribute used to allocate arrays with intent(out).
  Use the mold attribute to define shape of allocated array.

### Fixed
- Add continuations to long lines.
  This helps Fortran which has a line length limit.
- Write generic interface for overloaded constructors.

### Changed
- Attributes for a function now go at the end.
  void Foo+len=30 ()   =>  void Foo() +len(30)
  Fixes an ambiguity in attributes without values: Foo +attr(int i)
- Change prefix of local variables to SHC_ or SHCXX_ to indicate their usage.
- Added more local variables to convert between languages instead of doing
  conversion in-line  (in the call or return statements)

## v0.7.0 - 2018-01-22
### Added
- cpp_if will add an #if directive around a class or function.
- Allow 'functions' to be used in YAML in place of 'methods'.
  'methods' still works but only one should be provided.
- Allow format fields to be set by directly in YAML.

### Fixed
- Allow `std:string *` arguments and results.

### Changed
- All cpp_ prefixes for options and fmt have changed to cxx_.
- Moved many options and fields into format.  This eliminates
  copying options and fields into format and makes it clearer
  where to set formats.

### Fixed
- Respect wrap_* options for classes.

## v0.6.0 - 2018-01-10
### Changed
- Improved API for creating LibraryNode instances by using keyword arguments.
- Factored out routine create_library_from_dictionary for dealing with YAML
  generated dictionary.
- Moved code into generate.py which generates additional routines to wrap.

## v0.5.0 - 2018-01-09
### Added
- File shroud/ast.py with LibraryNode, ClassNode, and FunctionNode.

## v0.4.0 - 2018-01-05
### Added
- Recursive Descent Parser to replace Parsley.
- Parse `long long`.
- Parse constructors and destructors using C++ syntax.
  `Class * new() +constructor` changed to `Class() +name(new)`
- Ability to wrap pure C library.
  `language: c` at the top level of YAML file.

### Removed
- Parsley dependency.

## v0.3.0 - 2017-12-10
### Added
- Support for std::vector.
- Ability to set Filename suffixes with an option.
  `C_header_filename_suffix`, `C_impl_filename_suffix`
  `F_filename_suffix`,
  `PY_header_filename_suffix`, `PY_impl_filename_suffix`
  `LUA_header_filename_suffix`, `LUA_impl_filename_suffix`

### Removed
- Do not create `shroudrt.cpp` and `shroudrt.h` file.
  Helper functions are added directly to wrapper files.

## v0.2.0 - 2017-10-26
### Initial Open Source Release
