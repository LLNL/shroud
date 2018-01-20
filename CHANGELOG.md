
# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](http://semver.org/spec/v2.0.0.html).

## [Unreleased]
## Added
- cpp_if will add an #if directive around a class or function.
- Allow 'functions' to be used in YAML in place of 'methods'.
  'methods' still works but only one should be provided. 

## Fixed
- Allow `std:string *` arguments and results.

## Changed
- All cpp_ prefixes for options and fmt have changed to cxx_.

## Fixed
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
