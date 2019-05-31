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
    If arg is an array, add the ``dimension(:)`` attribute.
    This will create an assumed-shape attribute for ``arg``.
    The C array needs to have some way to determine the length of the
    array.  One option is to add another argument which will pass
    the length of the array from Fortran - ``int larg+implied(size(arg))``.
    See example :ref:`Sum <example_Sum>`.

.. XXX pointers should result to inout

``int &min +intent(out)``
