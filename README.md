# Shroud: generate Fortran and Python wrappers for C and C++ libraries.

**Shroud** is a tool for creating a Fortran or Python interface to a C
or C++ library.  It can also create a C API for a C++ library.

The user creates a YAML file with the C/C++ declarations to be wrapped
along with some annotations to provide semantic information and code
generation options.  **Shroud** produces a wrapper for the library.
The generated code is highly-readable and intended to be similar to code
that would be hand-written to create the bindings.

verb
1. wrap or dress (a body) in a shroud for burial.
2. cover or envelop so as to conceal from view.

[![Build Status](https://travis-ci.org/LLNL/shroud.svg?branch=develop)](https://travis-ci.org/LLNL/shroud)
[![Documentation Status](https://readthedocs.org/projects/shroud/badge/?version=develop)](http://shroud.readthedocs.io/en/latest/?badge=develop)

## Documentation

To get started using Shroud, check out the full documentation:

http://shroud.readthedocs.io/en/develop

## Mailing List

shroud-users@llnl.gov

## Required Packages

*  yaml - https://pypi.python.org/pypi/PyYAML

## C++ to C to Fortran

The generated Fortran requires a Fortran 2003 compiler.

## C++ or C to Python

The generated Python requires Python 2.7 or 3.4+.

## Release

Copyright (c) 2017-2020, Lawrence Livermore National Security, LLC and
other Shroud Project Developers.
See the top-level COPYRIGHT file for details.

SPDX-License-Identifier: (BSD-3-Clause)

Unlimited Open Source - BSD 3-clause Distribution

For release details and restrictions, please read the LICENSE.txt file.
It is also linked here:
- [LICENSE](./LICENSE)

`LLNL-CODE-738041`  `OCEC-17-143`
