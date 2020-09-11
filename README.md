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

## Goals

- Simplify the creating of wrapper for a C++ library.
- Preserves the object-oriented style of C++ classes.
- Create an idiomatic wrapper API from the C++ API.
- Generate code which is easy to understand.
- No dependent runtime library.

## Example

The user creates a YAML file which includes declarations from `zoo.hpp`.

```
library: zoo
cxx_header: zoo.hpp

declarations:
- decl: class Animal
  declarations:
  - decl: Animal()
  - decl: void speak(const std::string &word)
```
This creates a Fortran interface which can be used as:

```
use zoo_mod
type(Animal) dog
dog = Animal()
dog%speak("woof")
```

And from Python

```
import zoo
dog = zoo.Animal()
dog.speak("woof")
```

## Documentation

To get started using Shroud, check out the full documentation:

http://shroud.readthedocs.io/en/develop

## Mailing List

shroud-users@groups.io

https://groups.io/g/shroud-users

## Required Packages

*  yaml - https://pypi.python.org/pypi/PyYAML

## C++ to C to Fortran

The generated Fortran requires a Fortran 2003 compiler.

## C++ or C to Python

The generated Python requires Python 2.7 or 3.4+.

Python features:

- Uses NumPy for arrays. Also able to use Python lists if NumPy is overkill.
- Uses extension type for classes.
- Creates readable source.

## Getting started

Shroud can be installed using pip

```
pip install llnl-shroud
```

This can be done in a virtual environment as

```
cd my_project_folder
virtualenv my_project
source my_project/bin/activate
pip install llnl-shroud
```

This assumes the bash shell. Source activate.csh for csh.

In addition, a file created by
[shiv](https://github.com/linkedin/shiv)
is available from the github release.
Shroud and PyYAML are bundled into a single executable which uses
the Python3 on your path.
Shiv requires Python 3.6+.

```
wget https://github.com/LLNL/shroud/archive/shroud-0.12.2.pyz
```


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

