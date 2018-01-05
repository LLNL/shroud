
# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](http://semver.org/spec/v2.0.0.html).

## [Unreleased]

## v0.4.0 - 2018-01-14
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
	