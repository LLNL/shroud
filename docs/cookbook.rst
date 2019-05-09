.. Copyright (c) 2019, Lawrence Livermore National Security, LLC. 
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

Cookbook
========

F_name_impl with fortran_generic
--------------------------------

Using the *F_name_impl* format string to explicitly name a Fortran
wrapper when the *fortran_generic* field is used may present some
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
        n:
        -  int
        -  long

Will generate the Fortran code

.. code-block:: fortran

    interface change
        module procedure change_int
        module procedure change_long
    end interface change