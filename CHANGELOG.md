
# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](http://semver.org/spec/v2.0.0.html).

## [Unreleased]
- Add support for C++ structs.
  Fortran creates a derived type with ``bind(C)``.
  Python uses NumPy to unpack fields of struct.
- Wrap member variables in classes.
  Fortran and C create getter and setter functions.
  In Python, create a descriptor for each member.

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
