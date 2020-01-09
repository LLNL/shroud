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

## License

Copyright (c) 2017-2020, Lawrence Livermore National Security, LLC.
Produced at the Lawrence Livermore National Laboratory.

SPDX-License-Identifier: (BSD-3-Clause)

See [LICENSE](./LICENSE) for details

Unlimited Open Source - BSD 3-clause Distribution
`LLNL-CODE-738041`  `OCEC-17-143`

SPDX usage
------------

Individual files contain SPDX tags instead of the full license text.
This enables machine processing of license information based on the SPDX
License Identifiers that are available here: https://spdx.org/licenses/

Files that are licensed as BSD 3-Clause contain the following
text in the license header:

SPDX-License-Identifier: (BSD-3-Clause)

External Packages
-------------------
Shroud bundles some of its external dependencies in its repository.  These
packages are covered by various permissive licenses.  A summary listing
follows.  See the license included with each package for full details.

[//]: # (Note: The spaces at the end of each line below add line breaks)

PackageName: fruit  
PackageHomePage: https://sourceforge.net/projects/fortranxunit/  
PackageLicenseDeclared: BSD-3-Clause  

