.. Copyright (c) 2017-2020, Lawrence Livermore National Security, LLC and
   other Shroud Project Developers.
   See the top-level COPYRIGHT file for details.

   SPDX-License-Identifier: (BSD-3-Clause)

Cookbook
========

Function is really a macro or function pointer
----------------------------------------------

When wrapping a C library, a function which is really a macro may not
create a C wrapper.  It is necessary to use the option
``C_force_wrapper: true`` to create a wrapper which will expand the
macro and create a function which the Fortran wrapper may call.  This
same issue occurs when the function is really a function pointer.

When wrapping C++, a C wrapper is always created to create a extern C
symbol that Fortran can call.  So this problem does not occur.

F_name_impl with fortran_generic
--------------------------------

Using the *F_name_impl* format string to explicitly name a Fortran
wrapper combined with the *fortran_generic* field may present some
surprising behavior.  The routine ``BA_change`` takes a ``long``
argument.  However, this is inconvenient in Fortran since the default
integer is typically an ``int``.  When passing a constant you need to
explicitly state the kind as ``0_C_LONG``. Shroud lets you create a
generic routine which will also accept ``0``.  But if you explicitly
name the function using *F_name_impl*, both Fortran generated
functions will have the same name.  The solution is to set format field
*F_name_generic* and the option for *F_name_impl_template*.

.. code-block:: yaml

    - decl: int BA_change(const char *name, long n)
      format:
        F_name_generic: change
      options:
        F_name_impl_template: "{F_name_generic}{function_suffix}"
      fortran_generic:
      - decl: (int n)
        function_suffix: int
      - decl: (long n)
        function_suffix: long

Will generate the Fortran code

.. code-block:: fortran

    interface change
        module procedure change_int
        module procedure change_long
    end interface change
