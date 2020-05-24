.. Copyright (c) 2017-2020, Lawrence Livermore National Security, LLC and
   other Shroud Project Developers.
   See the top-level COPYRIGHT file for details.

   SPDX-License-Identifier: (BSD-3-Clause)

.. _TypesAnchor:

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



Numeric Types
--------------

The numeric types usually require no conversion.
In this case the type map is mainly used to generate declaration code 
for wrappers:

.. code-block:: yaml

    type: int
    fields:
        c_type: int 
        cxx_type: int
        f_type: integer(C_INT)
        f_kind: C_INT
        f_module:
            iso_c_binding:
            - C_INT
        f_cast: int({f_var}, C_INT)

One case where a conversion is required is when the Fortran argument
is one type and the C++ argument is another. This may happen when an
overloaded function is generated so that a ``C_INT`` or ``C_LONG``
argument may be passed to a C++ function function expecting a
``long``.  The **f_cast** field is used to convert the argument to the
type expected by the C++ function.


Bool
----


The first thing to notice is that **f_c_type** is defined.  This is
the type used in the Fortran interface for the C wrapper.  The type
is ``logical(C_BOOL)`` while **f_type**, the type of the Fortran
wrapper argument, is ``logical``.

The **f_statements** section describes code to add into the Fortran
wrapper to perform the conversion.  *c_var* and *f_var* default to
the same value as the argument name.  By setting **c_local_var**, a
local variable is generated for the call to the C wrapper.  It will be
named ``SH_{f_var}``.

There is no Fortran intrinsic function to convert between default
``logical`` and ``logical(C_BOOL)``. The **pre_call** and
**post_call** sections will insert an assignment statement to allow
the compiler to do the conversion.


If a function returns a ``bool`` result then a wrapper is always needed
to convert the result.  The **result** section sets **need_wrapper**
to force the wrapper to be created.  By default a function with no
argument would not need a wrapper since there will be no **pre_call**
or **post_call** code blocks.  Only the C interface would be required
since Fortran could call the C function directly.

See example :ref:`checkBool <example_checkBool>`.

Char
----

..  It also helps support ``const`` vs non-``const`` strings.

Any C++ function which has ``char`` or ``std::string`` arguments or
result will create an additional C function which include additional
arguments for the length of the strings.  Most Fortran compiler use
this convention when passing ``CHARACTER`` arguments. Shroud makes
this convention explicit for three reasons:

* It allows an interface to be used.  Functions with an interface will
  not pass the hidden, non-standard length argument, depending on compiler.
* It may pass the result of ``len`` and/or ``len_trim``.
  The convention just passes the length.
* Returning character argument from C to Fortran is non-portable.

Arguments with the *intent(in)* annotation are given the *len_trim*
annotation.  The assumption is that the trailing blanks are not part
of the data but only padding.  Return values and *intent(out)*
arguments add a *len* annotation with the assumption that the wrapper
will copy the result and blank fill the argument so it need to know
the declared length.

The additional function will be named the same as the original
function with the option **C_bufferify_suffix** appended to the end.
The Fortran wrapper will use the original function name, but call the
C function which accepts the length arguments.

The character type maps use the **c_statements** section to define
code which will be inserted into the C wrapper. *intent_in*,
*intent_out*, and *result* subsections add actions for the C wrapper.
*intent_in_buf*, *intent_out_buf*, and *result_buf* are used for
arguments with the *len* and *len_trim* annotations in the additional
C wrapper.

There are occasions when the *bufferify* wrapper is not needed.  For
example, when using ``char *`` to pass a large buffer.  It is better
to just pass the address of the argument instead of creating a copy
and appending a ``NULL``.  The **F_create_bufferify_function** options
can set to *false* to turn off this feature.


Char
^^^^



``Ndest`` is the declared length of argument ``dest`` and ``Lsrc`` is
the trimmed length of argument ``src``.  These generated names must
not conflict with any other arguments.  There are two ways to set the
names.  First by using the options **C_var_len_template** and
**C_var_trim_template**. This can be used to control how the names are
generated for all functions if set globally or just a single function
if set in the function's options.  The other is by explicitly setting
the *len* and *len_trim* annotations which only effect a single
declaration.

The pre_call code creates space for the C strings by allocating
buffers with space for an additional character (the ``NULL``).  The
*intent(in)* string copies the data and adds an explicit terminating
``NULL``.  The function is called then the post_call section copies
the result back into the ``dest`` argument and deletes the scratch
space.  ``ShroudStrCopy`` is a function provided by Shroud which
copies character into the destination up to ``Ndest`` characters, then
blank fills any remaining space.


MPI_Comm
--------

MPI_Comm is provided by Shroud and serves as an example of how to wrap
a non-native type.  MPI provides a Fortran interface and the ability
to convert MPI_comm between Fortran and C. The type map tells Shroud
how to use these routines:

.. code-block:: yaml

        type: MPI_Comm
        fields:
            cxx_type: MPI_Comm
            c_header: mpi.h
            c_type: MPI_Fint
            f_type: integer
            f_kind: C_INT
            f_c_type: integer(C_INT)
            f_c_module:
                iso_c_binding:
                  - C_INT
            cxx_to_c: MPI_Comm_c2f({cxx_var})
            c_to_cxx: MPI_Comm_f2c({c_var})


This mapping makes the assumption that ``integer`` and
``integer(C_INT)`` are the same type.


.. Complex Type
   ------------

Templates
---------

Shroud will wrap templated classes and functions for explicit instantiations.
The template is given as part of the ``decl`` and the instantations are listed in the
``cxx_template`` section:

.. code-block:: yaml

  - decl: |
        template<typename ArgType>
        void Function7(ArgType arg)
    cxx_template:
    - instantiation: <int>
    - instantiation: <double>

``options`` and ``format`` may be provide to control the generated code:

.. code-block:: yaml

  - decl: template<typename T> class vector
    cxx_header: <vector>
    cxx_template:
    - instantiation: <int>
      format:
        C_impl_filename: wrapvectorforint.cpp
      options:
        optblah: two
    - instantiation: <double>

.. from templates.yaml

For a class template, the *class_name* is modified to included the
instantion type.  If only a single template parameter is provided,
then the template argument is used.  For the above example,
*C_impl_filename* will default to ``wrapvector_int.cpp`` but has been
explicitly changed to ``wrapvectorforint.cpp``.

