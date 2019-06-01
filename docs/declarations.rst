.. Copyright (c) 2017-2019, Lawrence Livermore National Security, LLC. 
..
.. Produced at the Lawrence Livermore National Laboratory 
..
.. LLNL-CODE-738041.
..
.. All rights reserved. 
..
.. This file is part of Shroud.
..
.. For details about use and distribution, please read LICENSE.
..
.. #######################################################################

.. _DeclarationsAnchor:

Declarations
============

In order for Shroud to create an idiomatic wrapper, it needs to know
how arguments are intended to be used.  This information is supplied
via attributes. This section describes how to describe the arguments
to Shroud in order to implement the desired semantic.

Numeric Types
-------------

Integer and floating point numbers are supported by the
*interoperabilty with C* feature of Fortran 2003.  This includes the
integer types ``short``, ``int``, ``long`` and ``long long``.
Size specific types ``int8_t``, ``int16_t``, ``int32_t``, and
``int64_t`` are supported.
Floating point types ``float`` and ``double``.

.. note::  Fortran has no support for unsigned types.
           ``size_t`` will be the correct number of bytes, but
           will be signed.

In the following examples, ``int`` can be replaced by any numeric type.

``int arg``
    Pass a numeric value to C.  The attribute ``intent(in)`` is defaulted.
    The Fortran 2003 attribute ``VALUE`` is used to change from
    Fortran's default call-by-reference to C's call-by-value.
    This argument can be called directly by Fortran and no C wrapper is 
    necessary.
    See example :ref:`PassByValue <example_PassByValue>`.

``int *arg``
    If the intent is to return a scalar value from a function,
    add the ``intent(out)`` attribute.
    See example :ref:`PassByReference <example_PassByReference>`.

``int *arg``
    If ``arg`` is an array, add the ``dimension(:)`` attribute.
    This will create an assumed-shape attribute for ``arg``.
    The C array needs to have some way to determine the length of the
    array.  One option is to add another argument which will pass
    the length of the array from Fortran - ``int larg+implied(size(arg))``.
    See example :ref:`Sum <example_Sum>`.

.. XXX pointers should result to inout

``int &min +intent(out)``
    A declaration to a scalar gets converted into pointers in the
    C wrapper.
    See example :ref:`getMinMax <example_getMinMax>`.

Bool
----

C and C++ functions with a ``bool`` argument generate a Fortran wrapper with
a ``logical`` argument.  One of the goals of Shroud is to produce an
idiomatic interface.  Converting the types in the wrapper avoids the
awkwardness of requiring the Fortran user to passing in
``.true._c_bool`` instead of just ``.true.``.


``bool arg``
    Non-pointer arguments default to ``intent(IN)``.
    See example :ref:`checkBool <example_checkBool>`.

Char
----

Fortran, C, and C++ each have their own semantics for character variables.

  * Fortran ``character`` variables know their length and are blank filled
  * C ``char *`` variables are assumed to be ``NULL`` terminated.
  * C++ ``std::string`` know their own length and can provide a ``NULL`` terminated pointer.

It is not sufficient to pass an address between Fortran and C++ like
it is with other native types.  In order to get idiomatic behavior in
the Fortran wrappers it is often necessary to copy the values.  This
is to account for blank filled vs ``NULL`` terminated.


``const char *arg``
    Create a ``NULL`` terminated string in Fortran using
    ``trim(arg)//C_NULL_CHAR`` and pass to C.
    Since the argument is ``const``, it is treated as ``intent(in)``.
    A bufferify function is not required to convert the argument.
    This is the same as ``char *arg+intent(in)``.
    See example :ref:`acceptName <example_acceptName>`.

``char *arg``
    Pass a ``char`` pointer to a function which assign to the memory.
    ``arg`` must be ``NULL`` terminated by the function.
    Add the *intent(out)* attribute.
    The bufferify function will then blank-fill the string to the length
    of the Fortran ``CHARACTER(*)`` argument.
    It is the users responsibility to avoid overwriting the argument. 
    See example :ref:`returnOneName <example_returnOneName>`.

    Fortran must provide a CHARACTER argument which is at least as long as
    the amount that the C function will write into.  This includes space
    for the terminating NULL which will be converted into a blank for
    Fortran.

``char *arg, int larg``
    Similar to above, but pass in the length of ``arg``.
    The argument ``larg`` does not need to be passed to Fortran explicitly
    since its value is implied.
    The *implied* attribute is defined to use the ``len`` Fortran 
    intrinsic to pass the length of ``arg`` as the value of ``larg``:
    ``char *arg+intent(out), int larg+implied(len(arg))``.
    See example :ref:`ImpliedTextLen <example_ImpliedTextLen>`.

std::string
-----------


char functions
--------------

Functions which return a ``char *`` provide an additional challenge.
Taken literally they should return a ``type(C_PTR)``.  And if you call
the function via the interface, that's what you get.  However,
Shroud provides several options to provide a more idiomatic usage.

Each of these declaration call identical C++ functions but they are
wrapped differently

``char *getCharPtr1``

    Return a pointer and convert into an ``ALLOCATABLE`` ``CHARACTER``
    variable.  Fortran 2003 is required. The Fortran application is
    responsible to release the memory.  However, this may be done
    automatically by the Fortran runtime.

    See example :ref:`getCharPtr1 <example_getCharPtr1>`.

``char *getCharPtr2``

    Create a Fortran function which returns a predefined ``CHARACTER`` 
    value.  The size is determined by the *len* argument on the function.
    This is useful when the maximum size is already known.
    Works with Fortran 90.

    See example :ref:`getCharPtr2 <example_getCharPtr2>`.

``char *getCharPtr3``

    Create a Fortran subroutine in an additional ``CHARACTER``
    argument for the C function result. Any size character string can
    be returned limited by the size of the Fortran argument.  The
    argument is defined by the *F_string_result_as_arg* format string.
    Works with Fortran 90.

    See example :ref:`getCharPtr3 <example_getCharPtr3>`.


