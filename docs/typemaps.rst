.. Copyright (c) 2017-2020, Lawrence Livermore National Security, LLC and
   other Shroud Project Developers.
   See the top-level COPYRIGHT file for details.

   SPDX-License-Identifier: (BSD-3-Clause)

.. _TypemapsAnchor:

Typemaps
========

A typemap is created for each type to describe to Shroud how it should
convert a type between languages for each wrapper.  Native types are
predefined and a Shroud typemap is created for each ``struct`` and
``class`` declaration.

The general form is:

.. code-block:: yaml

    declarations:
    - type: type-name
      fields:
         field1:
         field2:

*type-name* is the name used by C++.  There are some fields which are
used by all wrappers and other fields which are used by language
specific wrappers.

type fields
-----------

These fields are common to all wrapper languages.

base
^^^^

The base type of *type-name*.
This is used to generalize operations for several types.
The base types that Shroud uses are **string**, **vector**, 
or **shadow**.

cpp_if
^^^^^^

A c preprocessor test which is used to conditionally use
other fields of the type such as *c_header* and *cxx_header*:

.. code-block:: yaml

  - type: MPI_Comm
    fields:
      cpp_if: ifdef USE_MPI

flat_name
^^^^^^^^^

A flattened version of **cxx_type** which allows the name to be 
used as a legal identifier in C, Fortran and Python.
By default any scope separators are converted to underscores
i.e. ``internal::Worker`` becomes ``internal_Worker``.
Imbedded blanks are converted to underscores
i.e. ``unsigned int`` becomes ``unsigned_int``.
And template arguments are converted to underscores with the trailing
``>`` being replaced
i.e. ``std::vector<int>`` becomes ``std_vector_int``.

One use of this name is as the **function_suffix** for templated functions.

idtor
^^^^^

Index of ``capsule_data`` destructor in the function
*C_memory_dtor_function*.
This value is computed by Shroud and should not be set.
It can be used when formatting statements as ``{idtor}``.
Defaults to *0* indicating no destructor.

.. format field

C and C++
---------

c_type
^^^^^^

Name of type in C.
Default to *None*.


c_header
^^^^^^^^

Name of C header file required for type.
This file is included in the interface header.
Only used with *language=c*.
Defaults to *None*.

See also *cxx_header*.


c_to_cxx
^^^^^^^^

Expression to convert from C to C++.
Defaults to *None* which implies *{c_var}*.
i.e. no conversion required.


c_templates
^^^^^^^^^^^

c_statements for cxx_T

A dictionary indexed by type of specialized *c_statements* When an
argument has a *template* field, such as type ``vector<string>``, some
additional specialization of c_statements may be required::

        c_templates:
            string:
               intent_in_buf:
               - code to copy CHARACTER to vector<string>



c_return_code
^^^^^^^^^^^^^

None

c_union
^^^^^^^

None
# Union of C++ and C type (used with structs and complex)

cxx_type
^^^^^^^^

Name of type in C++.
Defaults to *None*.


cxx_to_c
^^^^^^^^

Expression to convert from C++ to C.
Defaults to *None* which implies *{cxx_var}*.
i.e. no conversion required.

cxx_header
^^^^^^^^^^

Name of C++ header file required for implementation.
For example, if cxx_to_c was a function.
Only used with *language=c++*.
Defaults to *None*.
Note the use of *stdlib* which adds ``std::`` with *language=c++*:

.. code-block:: yaml

    c_header='<stdlib.h>',
    cxx_header='<cstdlib>',
    pre_call=[
        'char * {cxx_var} = (char *) {stdlib}malloc({c_var_len} + 1);',
    ],

See also *c_header*.

A C ``int`` is represented as:

.. code-block:: yaml

    type: int
    fields:
        c_type: int 
        cxx_type: int


Fortran
-------

f_c_module
^^^^^^^^^^

Fortran modules needed for type in the interface.
A dictionary keyed on the module name with the value being a list of symbols.
Similar to **f_module**.
Defaults to *None*.

f_c_type
^^^^^^^^

Type declaration for ``bind(C)`` interface.
Defaults to *None* which will then use *f_type*.

f_cast
^^^^^^

Expression to convert Fortran type to C type.
This is used when creating a Fortran generic functions which
accept several type but call a single C function which expects
a specific type.
For example, type ``int`` is defined as ``int({f_var}, C_INT)``.
This expression converts *f_var* to a ``integer(C_INT)``.
Defaults to *{f_var}*  i.e. no conversion.

..  See tutorial function9 for example.  f_cast is only used if the types are different.


f_derived_type
^^^^^^^^^^^^^^

Fortran derived type name.
Defaults to *None* which will use the C++ class name
for the Fortran derived type name.


f_kind
^^^^^^

Fortran kind of type. For example, ``C_INT`` or ``C_LONG``.
Defaults to *None*.


f_module
^^^^^^^^

Fortran modules needed for type in the implementation wrapper.  A
dictionary keyed on the module name with the value being a list of
symbols.
Defaults to *None*.:

.. code-block:: yaml

    f_module:
       iso_c_binding:
       - C_INT

f_type
^^^^^^

Name of type in Fortran.  ( ``integer(C_INT)`` )
Defaults to *None*.

f_to_c
^^^^^^

None
Expression to convert from Fortran to C.



example

An ``int`` argument is converted to Fortran with the typemap:

.. code-block:: yaml

    type: int
    fields:
        f_type: integer(C_INT)
        f_kind: C_INT
        f_module:
            iso_c_binding:
            - C_INT
        f_cast: int({f_var}, C_INT)





   

Statements
----------

Each language also provides a section that is used 
to insert language specific statements into the wrapper.
These are named **c_statements**, **f_statements**, and
**py_statements**.

The are broken down into several resolutions.  The first is the
intent of the argument.  *result* is used as the intent for 
function results.

in
    Code to add for argument with ``intent(IN)``.
    Can be used to convert types or copy-in semantics.
    For example, ``char *`` to ``std::string``.

out
    Code to add after call when ``intent(OUT)``.
    Used to implement copy-out semantics.

inout
    Code to add after call when ``intent(INOUT)``.
    Used to implement copy-out semantics.

result
    Result of function.
    Including when it is passed as an argument, *F_string_result_as_arg*.


Each intent is then broken down into code to be added into
specific sections of the wrapper.  For example, **declaration**,
**pre_call** and **post_call**.

Each statement is formatted using the format dictionary for the argument.
This will define several variables.

c_var
    The C name of the argument.

cxx_var
    Name of the C++ variable.

f_var
    Fortran variable name for argument.

For example:

.. code-block:: yaml

    f_statements:
      intent_in:
      - '{c_var} = {f_var}  ! coerce to C_BOOL'
      intent_out:
      - '{f_var} = {c_var}  ! coerce to logical'

Note that the code lines are quoted since they begin with a curly brace.
Otherwise YAML would interpret them as a dictionary.

See the language specific sections for details.



