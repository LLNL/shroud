.. Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
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
The base types that Shroud uses are **bool**, **integer**, **real**,
**complex**, **string**, **vector**, **struct** or **shadow**.


**integer** includes all integer types such as ``short``, ``int`` and ``long``.

.. **template**

.. used with Fortran declaration:  {base}({kind})

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

Complex types set this explicitly since C and C++ have much different
type names. The *flat_name* is always ``double_complex`` while
*c_type* is ``double complex`` and *cxx_type* is ``complex<double>``.


One use of this name is as the **function_suffix** for templated functions.

implied_array
^^^^^^^^^^^^^

The type is an implied array. For example, ``std::vector``.
It is not a pointer type but is considered to be an array.
This will set the default *deref* attribute based on the option
**F_implied_array**.

idtor
^^^^^

Index of ``capsule_data`` destructor in the function
*C_memory_dtor_function*.
This value is computed by Shroud and should not be set.
It can be used when formatting statements as ``{idtor}``.
Defaults to *0* indicating no destructor.

sgroup
^^^^^^

Groups different base types together.
For example, base *integer* and *real* are both sgroup *native*.
For many others, they're the same: base=struct, sgroup=struct.

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

For example, ``size_t`` requires stddef.h:

.. code-block:: yaml

    type: size_t
    fields:
        c_type: size_t 
        cxx_type: size_t
        c_header: <stddef.h>


c_to_cxx
^^^^^^^^

Expression to convert from C to C++.
Defaults to *None* which implies *{c_var}*.
i.e. no conversion required.

For typedefs, this will use a ``static_cast`` to convert
between equivelent types.

See also *cxx_to_c*.

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


cxx_type
^^^^^^^^

Name of type in C++.
Defaults to *None*.


cxx_to_c
^^^^^^^^

Expression to convert from C++ to C.
Defaults to *None* which implies *{cxx_var}*.
i.e. no conversion required.

Native POD types do not require any conversion.
The ``std::string`` uses the ``c_str`` method to get a ``char *``.

See also *c_to_cxx*.

cxx_header
^^^^^^^^^^

Name of C++ header file required for implementation.


.. For example, if cxx_to_c was a function.
   Only used with *language=c++*.
   Defaults to *None*.
   Note the use of *stdlib* which adds ``std::`` with *language=c++*:

.. code-block:: yaml

    c_type: size_t
    c_header: '<stddef.h>'
    cxx_header: '<cstddef>'

See also *c_header*.

impl_header
^^^^^^^^^^^

**impl_header** is used for implementation, i.e. the ``wrap.cpp`` file.
For example, ``std::string`` uses ``<string>``.
``<string>`` should not be in the interface since the wrapper is a C API.


wrap_header
^^^^^^^^^^^

**wrap_header** is used for generated wrappers for shadow classes.
Contains struct definitions for capsules from Fortran.

.. ---------------------

A C ``int`` is represented as:

.. code-block:: yaml

    type: int
    fields:
        c_type: int 
        cxx_type: int


Fortran
-------

f_type
^^^^^^

Name of type in Fortran.
For example, ``integer(C_INT)``.

f_kind
^^^^^^

Fortran kind of type. For example, ``C_INT`` or ``C_LONG``.
It will be set the same as *f_derived_type* for derived types.
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

f_derived_type
^^^^^^^^^^^^^^

Fortran derived type name.
Defaults to *None* which will use the C++ class name
for the Fortran derived type name.

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

f_to_c
^^^^^^

None
Expression to convert from Fortran to C.



example

An ``int`` argument is converted to Fortran with the typemap:

.. code-block:: yaml

    typemap:
    - type: int
      fields:
          f_type: integer(C_INT)
          f_kind: C_INT
          f_module:
              iso_c_binding:
              - C_INT
          f_cast: int({f_var}, C_INT)

.. Example from forward.yaml...
          
A ``struct`` defined in another YAML file.

.. code-block:: yaml

    typemap:
    - type: Cstruct1
      fields:
        base: struct
        cxx_header:
        - struct.hpp
        wrap_header:
        - wrapstruct.h
        c_type: STR_cstruct1
        f_derived_type: cstruct1
        f_module_name: struct_mod
                
.. XXX - explain about generated type file.
   
Also used to extract the *F_derived_member* before passing to C.

ci_type
^^^^^^^

The type of the argument in the C bufferify wrapper.
Usually this is the same as *c_type*.

One case where it is different is with enumerations.  In Fortran, the
option *F_enum_type" determines the type of ``enum`` values in
Fortran. This defaults to an ``int`` which is then cast to the correct
type using *c_to_cxx*.

.. ci_to_cxx does not exist since it would always be the samea as c_to_cxx.

cxx_to_ci
^^^^^^^^^

Convert the C++ type into a Fortran interface type.
Used to convert function return values.
If unset, then *cxx_to_c* is used.

Interface
---------

i_type
^^^^^^

Type declaration for ``bind(C)`` interface.
For example, ``integer(C_INT)``.

.. Defaults to *None* which will then use *f_type*.


i_module_name
^^^^^^^^^^^^^

Name of module required for interface type.
For example, ``iso_c_binding``.


i_kind
^^^^^^

Kind parameter required for interface type.
For example, ``C_INT``.

i_module
^^^^^^^^

Fortran modules needed for type in the interface.
A dictionary keyed on the module name with the value being a list of symbols.
Similar to **f_module**.
Defaults to *None*.

Examples
--------

Fortran native types are used for ``LOGICAL`` and ``CHARACTER``.
So *f_kind* and *f_module_name* are not defined.
But for the interface, *i_kind* and *i_module_name* are defined.

.. code-block:: yaml

    bool:
       f_type: logical
       f_kind: ""
       f_module_name: ""

       i_type: logical(C_BOOL)
       i_kind: C_BOOL
       i_module_name: iso_c_binding


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



