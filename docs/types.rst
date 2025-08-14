.. Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
   other Shroud Project Developers.
   See the top-level COPYRIGHT file for details.

   SPDX-License-Identifier: (BSD-3-Clause)

Types
=====

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

The first thing to notice is that **i_type** is defined.  This is
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
this convention explicit for two reasons

    * It allows an interface to be used.  Functions with an interface will
      not pass the hidden, non-standard length argument, depending on compiler.

    * Returning character argument from C to Fortran is non-portable.
      Often an additional argument is added for the function result.

The C wrapper will create a NULL terminated copy a string with the
*intent(in)* attribute.  The assumption is that the trailing blanks
are not part of the data but only padding.  Return values and
*intent(out)* arguments add a *len* annotation with the assumption
that the wrapper will copy the result and blank fill the argument so
it need to know the declared length.

A buffer for *intent(out)* arguments is also create which is one
longer than the Fortran string length. This allows space for a C
terminating NULL. This buffer is passed to the C library which will
copy into it.  Upon return, the buffer is copied and blank filled into
the user's argument and the intermediate buffer released.

Library functions which return a scalar ``char`` have a wrapper generated
which pass a ``char *`` argument to the C wrapper where the first
element is assigned ( ``*arg`` a.k.a ``arg[0]``). Returning a ``char``
proved to be non-portable while passing the result by reference works
on the tested compilers.

The bufferify function will be named the same as the original
function with the option **C_bufferify_suffix** appended to the end.
The Fortran wrapper will use the original function name, but call the
C wrapper which accepts the length arguments.

Python wrappers may need an additional attribute for *intent(out)*
strings to let Shroud know how much space to pass to the function. A
function may pass a ``char *`` argument which the C library will copy
into.  While this is not a recommened practice since it's easy to
overwrite memory, Shoud can deal with it by setting the *+charlen(n)*
attribute where *n* is the number of character in the array passed to
the function. This is required for Python since strings are inmutable.
The buffer will be converted into a Python str object then returned to
the user. This is not an issue in Fortran since the output buffer is
passed in by the caller and will have a known size.

By default, a Fortran blank input string will be converted to an empty
string before being passed to the C library.  i.e. ``" "`` in Fortran
is converted to ``'\0'`` in C. This behavior can be changed to convert
the empty string into a ``NULL`` pointer by setting the *+blanknull*
attribute. This is often more natural for the C library to indicate the
absence of a value. The option *F_blanknull* can be used to make this the
default for all ``const char *`` arguments.

On some occasions the copy and null terminate behavior is not wanted.
For example, to avoid copying a large buffer or the memory must be
operated on directly.  In this case using the attribute *+api(capi)*
will use the native C API instead of the bufferify API for the
argument.  The library will need some way to determine the length of
the string since it will not be passed to the C wrapper.  As an
alternative the bufferify function can be avoided altogether by
setting the **F_create_bufferify_function** option to *false*.


The character type maps use the **c_statements** section to define
code which will be inserted into the C wrapper.  These actions vary
depending on the intent of *in*, *out*, *inout* and *result*.

.. option F_trim_char_in

.. ``Ndest`` is the declared length of argument ``dest`` and ``Lsrc``
   is the trimmed length of argument ``src``.  These generated names must
   not conflict with any other arguments.  There are two ways to set the
   names.  First by using the options **C_var_len_template** and
   **C_var_trim_template**. This can be used to control how the names are
   generated for all functions if set globally or just a single function
   if set in the function's options.  The other is by explicitly setting
   the *len* and *len_trim* annotations which only effect a single
   declaration.


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
            i_type: integer(C_INT)
            i_module:
                iso_c_binding:
                  - C_INT
            cxx_to_c: MPI_Comm_c2f({cxx_var})
            c_to_cxx: MPI_Comm_f2c({c_var})


This mapping makes the assumption that ``integer`` and
``integer(C_INT)`` are the same type.


.. Complex Type
   ------------


Typedef
-------

A typedef is used to create an alias for another type.
Often to create an abstraction for the intented use of the type.

.. code-block:: c++

   typedef int IndexType;

From the Fortran side, this will create a parameter for the kind parameter.

.. code-block:: fortran

   integer, parameter :: index_type = C_INT

This allows variables to continue to use the typedef.

.. code-block:: c++

   IndexType arg;

From the Fortran side, this will create a parameter for the kind parameter.

.. code-block:: fortran

   integer(index_type) :: arg

A C wrapper will create another typedef which is accessible from C.
A C++ typedef may be in a scope so Shroud will mangle the name
to make it accessible at the global level.

.. code-block:: c

    typedef int TYP_IndexType;

The name of the generated typedef can be modified using format fields.

.. code-block:: yaml

    - decl: typedef int32_t IndexType2
      format:
        F_name_typedef: LOCAL_Index_Type
        C_name_typedef: LOCAL_IndexType

.. from typedefs.yaml
