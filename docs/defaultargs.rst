.. Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and
   other Shroud Project Developers.
   See the top-level COPYRIGHT file for details.

   SPDX-License-Identifier: (BSD-3-Clause)

.. _DefaultArguments:

Default Arguments
=================

Default arguments allows a C++ function to be called without providing
one or more trailing arguments. Shroud can handle default args in
several different ways based on the value of option *F_default_args*.

Generic Default Arguments
-------------------------

Since a default argument can have any C++ value, the C++ compiler must
be used to provide the values. Shroud does this by creating a function
for each possible way to call the function. These functions are then
combined into a generic interface with the C++ function name.
This is the default behavior of Shroud but can be made explicit
by setting option *F_default_args* to *generic*.

For example, the function

.. code-block:: c++

    void apply(IndexType num_elems, IndexType offset = 0, IndexType stride = 1);

can be called with 1, 2 or 3 arguments. C wrapper functions are
created with the prototypes:

.. code-block:: c++

    void apply(IndexType num_elems);
    void apply(IndexType num_elems, IndexType offset);
    void apply(IndexType num_elems, IndexType offset, IndexType stride);

The C++ compiler will provided the missing arguments using the default
values.

The generated functions will have the same name as the C++ function
with a suffix added to create unique names.  By default this is a
integer sequence number. The suffix can be controlled by adding a
**default_arg_suffix** entry to the YAML file. One suffix is provided
for each generated overloaded function.

.. code-block:: yaml

    - decl: void apply(IndexType num_elems, IndexType offset = 0, IndexType stride = 1);
      default_arg_suffix:
      -  _nelems
      -  _nelems_offset
      -  _nelems_offset_stride



Require Default Arguments
-------------------------

Shroud provides the option to require all arguments by setting
*F_default_args* to *require*.  This is intended to help when there
are overloaded functions with default arguments.  The Fortran type
system is not a rich as C++ and some Fortran generic function may be
ambiguous. This can happen since C++ ``enum`` is converted to an
``integer``.


Optional Default Arguments
--------------------------

When the default values can be represented in Fortran the ``OPTIONAL``
attribute can be used with default arguments to allow the Fortran
wrapper to supply the value for arguments which are not present in the
call to the function.  This is generated when the option
*F_default_args* is set to *optional*.
No overloaded functions are generated.
The C wrapper will require all arguments to be provided.

This provides the ability to call the function from Fortran in a way
not supported by C++.  Each argument can be provided individually
using keyword arguments.  The function can be called as

.. code-block:: fortran

    call apply(100, stride=2)

and ``offset`` will be provided the default value of ``0``.

.. XXX Python also support keyword arguments.

Since the value is provided by Fortran, this only works with
integer and real values.

.. XXX logical/bool and strings are not working yet.

