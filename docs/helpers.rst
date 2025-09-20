.. Copyright Shroud Project Developers. See LICENSE file for details.

   SPDX-License-Identifier: (BSD-3-Clause)

.. _HelpersAnchor:

Helpers
=======

Helper functions are used to encapsulate functionality that may be
needed by many wrappers. This keeps the individual wrappers simpler
while sharing code among wrappers. Helpers may be in C or
Fortran. They can also provide a Fortran interface to a C function.

Helper code can be inserted in several files and several locations.

.. code-block:: c++

    // file C_header_util

    // scope=cwrap_include include

    #ifdef __cplusplus
    extern "C" {
    #endif

    // api=c scope=cwrap_include source

    // api=c proto

    #ifdef __cplusplus
    }
    #endif

    // api=c++ proto_include
    // api=c++ proto


.. code-block:: c++

    // file C_impl_util

    #ifdef __cplusplus
    extern "C" {
    #endif

    // api=c scope=cwrap_impl source

    #ifdef __cplusplus
    }
    #endif

    // api=c++ scope=cwrap_impl source


.. code-block:: c++

    // wrapper implementation file

    // scope=cwrap_include include

    // api=cxx scope=file source

    extern "C" {
    // api=c scope=file source

.. code-block:: fortran

    module name
                
    ! modules
    use module, only : symbol
                
    ! derived_type

    ! interface

    contains

    ! f_source

    end module name
                
.. helper_summary

Listed in the statements.helper field.
The C helpers are written after creating the Fortran wrappers by 
clibrary.write_impl_utility function.

The helpers will be preprocessed using format to expand some fields.
All helpers use the same dictionary. This allows a helper to add a field
using *fmtdict* which can be used by later helpers.
The format dictionary's parent is the library's format dictionary.
This give the helpers access to global names such as *C_array_type* and
*F_array_type*.

Fields which are formatted will need to follow the formatting rules.
The most common is that  ``{{`` is required to insert a literal ``{``.
Otherwise, it is assumed to be a format field.


Fields
------

name
^^^^

The name must start with ``h_helper_``.  The remainder of the string is used
as the helper name in the *helper* fields of statement groups.

.. The 'language' is ``h`` which defines the default values of fields.
   The 'intent' is ``helper``.

c_fmtname / f_fmtname
^^^^^^^^^^^^^^^^^^^^^

Name of function or type created by the helper.  This allows the
function name to be independent of the helper name so that it may
include a prefix to help control namespace/scope.  Useful when two
helpers create the same function.
Added to the wrapper's
format dictionary to allow it to be used in statements.

This field is formatted.

fmtdict
^^^^^^^

A dictionary which will be formatted and added to the common format dictionary
used to preprocess the helpers.

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

source
^^^^^^

List of lines of code inserted before any wrappers.
The functions should be file static.
Used if *c_source* or *cxx_source* is not defined.
This field is formatted.

c_source
^^^^^^^^

Lines of C code used when *language=c*.
This field is formatted.

cxx_source
^^^^^^^^^^

Lines of C++ code used when *language=c++*.
This field is formatted.

dependent_helpers
^^^^^^^^^^^^^^^^^

List of helpers names needed by this helper. They will be added to the
output before current helper.

proto
^^^^^

Prototype for helper function. Must be in the language of *api*.
This field is formatted.

proto_include
^^^^^^^^^^^^^

List of files to ``#include`` before the prototype.

Include in the utility file defined by format field *C_header_utility*.

include
^^^^^^^

List of files to ``#include``.
Used when *c_header* and *cxx_header* are not defined.


derived_type
^^^^^^^^^^^^

List of lines of Fortran code for a derived type.
Inserted before interfaces.
This field is formatted.

.. private   = names for PRIVATE statement

interface
^^^^^^^^^

List of lines of Fortran code for ``INTERFACE``.
This field is formatted.

f_source
^^^^^^^^

List of lines of Fortran code for ``CONTAINS``.
This field is formatted.

