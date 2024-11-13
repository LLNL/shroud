.. Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and
   other Shroud Project Developers.
   See the top-level COPYRIGHT file for details.

   SPDX-License-Identifier: (BSD-3-Clause)

.. shroud documentation master file, created by
   sphinx-quickstart on Sat Jul 11 12:50:59 2015.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Shroud
======

**Shroud** is a tool for creating a Fortran or Python interface to a C
or C++ library.  It can also create a C API for a C++ library.

The user creates a YAML file with the C/C++ declarations to be wrapped
along with some annotations to provide semantic information and code
generation options.  **Shroud** produces a wrapper for the library.
The generated code is highly-readable and intended to be similar to code
that would be hand-written to create the bindings.

*verb*
    1. wrap or dress (a body) in a shroud for burial.
    2. cover or envelop so as to conceal from view.


Contents

.. toctree::
   :maxdepth: 1

   introduction
   installing
   tutorial
   input
   pointers
   shared_ptr
   types
   namespaces
   struct
   defaultargs
   templates
   declarations
   preprocessing
   output
   cwrapper
   fortran
   python
   cookbook
   expand
   typemaps
   statements
   cstatements
   fstatements
   helpers
   reference
   releases
   previouswork
   pypreviouswork
   futurework
   appendix-A
   appendix-PY
   glossary


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

