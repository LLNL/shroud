.. Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and
   other Shroud Project Developers.
   See the top-level COPYRIGHT file for details.

   SPDX-License-Identifier: (BSD-3-Clause)

.. _HelpersAnchor:

Helpers
=======

Helper functions are used to encapsulate functionality that may be
needed by many wrappers. This keeps the individual wrappers simpler
while sharing code among wrappers. Helpers may be in C or
Fortran. They can also provide a Fortran interface to a C function.




Listed in the statements.c_helper and f_helper fields.
The C helpers are written after creating the Fortran wrappers by 
clibrary.write_impl_utility function.


Fields
------


c_fmtname / f_fmtname
^^^^^^^^^^^^^^^^^^^^^

Name of function or type created by the helper.  This allows the
function name to be independent of the helper name so that it may
include a prefix to help control namespace/scope.  Useful when two
helpers create the same function.
Added to the wrapper's
format dictionary to allow it to be used in statements.

api
^^^

``c`` or ``cxx``. Defaults to ``c``.
Must be set to ``c`` for helper functions which will be called from Fortran.
Helpers which use types such as ``std::string`` or ``std::vector``
can only be compiled with C++. Setting api to ``c`` will add 
the prototype in an ``extern "C"`` block.
Effects the fields *source*, *proto*, and *proto_include* are inserted.

scope
^^^^^
Scope of helper.

* ``file`` - (default) added as file static and may be in
  several files. source may set source, c_source, or cxx_source.
  functions must be static since they may be included in 
  multiple files.

* ``cwrap_include`` - Will add to C_header_utility and shared
  among files. These names need to be unique since they
  are shared across wrapped libraries.
  Used with structure and enums.

* ``cwrap_impl`` - Helpers which are written in C and called by C or Fortran.

c_include
^^^^^^^^^

List of files to ``#include`` in implementation file when wrapping a C library.

cxx_include
^^^^^^^^^^^

List of files to ``#include`` in implementation file when wrapping a C++ library.

c_source
^^^^^^^^

Lines of C code used when *language=c*.

cxx_source
^^^^^^^^^^

Lines of C++ code used when *language=c++*.

dependent_helpers
^^^^^^^^^^^^^^^^^

List of helpers names needed by this helper. They will be added to the
output before current helper.

proto
^^^^^

Prototype for helper function. Must be in the language of *api*.

proto_include
^^^^^^^^^^^^^

List of files to ``#include`` before the prototype.

source
^^^^^^

List of lines of code inserted before any wrappers.
The functions should be file static.
Used if *c_source* or *cxx_source* is not defined.

include
^^^^^^^

List of files to ``#include``.
Used when *c_header* and *cxx_header* are not defined.


derived_type
^^^^^^^^^^^^

List of lines of Fortran code for a derived type.
Inserted before interfaces.

.. private   = names for PRIVATE statement

interface
^^^^^^^^^

List of lines of Fortran code for ``INTERFACE``.

f_source
^^^^^^^^

List of lines of Fortran code for ``CONTAINS``.

