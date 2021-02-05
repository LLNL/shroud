.. Copyright (c) 2017-2021, Lawrence Livermore National Security, LLC and
   other Shroud Project Developers.
   See the top-level COPYRIGHT file for details.

   SPDX-License-Identifier: (BSD-3-Clause)


Fortran
=======

This section discusses Fortran specific wrapper details.
This will also include some C wrapper details since some C wrappers
are created specificially to be called by Fortran.

Wrapper
-------

As each function declaration is parsed a format dictionary is created
with fields to describe the function and its arguments.
The fields are then expanded into the function wrapper.

The template for Fortran code showing names which may 
be controlled directly by the input YAML file:

.. code-block:: text

    module {F_module_name}

      ! use_stmts
      implicit none

      abstract interface
         subprogram {F_abstract_interface_subprogram_template}
            type :: {F_abstract_interface_argument_template}
         end subprogram
      end interface

      interface
        {F_C_pure_clause} {F_C_subprogram} {F_C_name}
             {F_C_result_clause} bind(C, name="{C_name}")
          ! arg_f_use
          implicit none
          ! arg_c_decl
        end {F_C_subprogram} {F_C_name}
      end interface

      interface {F_name_generic}
        module procedure {F_name_impl}
      end interface {F_name_generic}

    contains

      {F_subprogram} {F_name_impl}
        decl_args
        declare      ! local variables
        pre_call
        call  {arg_c_call}
        post_call
      end {F_subprogram} {F_name_impl}

    end module {F_module_name}


Class
-----

Use of format fields for creating class wrappers.

.. code-block:: text

    type, bind(C) :: {F_capsule_data_type}
        type(C_PTR) :: addr = C_NULL_PTR  ! address of C++ memory
        integer(C_INT) :: idtor = 0       ! index of destructor
    end type {F_capsule_data_type}

    type {F_derived_name}
        type({F_capsule_data_type}) :: {F_derived_member}
    contains
        procedure :: {F_name_function} => {F_name_impl}
        generic :: {F_name_generic} => {F_name_function}, ...

        ! F_name_getter, F_name_setter, F_name_instance_get as underscore_name
        procedure :: [F_name_function_template] => [F_name_impl_template]

    end type {F_derived_name}


Standard type-bound procedures
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Several type bound procedures can be created to make it easier to 
use class from Fortran.

Usually the *F_derived_name* is constructed from wrapped C++
constructor.  It may also be useful to take a pointer to a C++ struct
and explicitly put it into a the derived type.  The functions
*F_name_instance_get* and *F_name_instance_set* can be used to access
the pointer directly.

.. Add methods to *F_capsule_data_type* directly?

Two predicate function are generated to compare derived types:

.. code-block:: text

        interface operator (.eq.)
            module procedure class1_eq
            module procedure singleton_eq
        end interface

        interface operator (.ne.)
            module procedure class1_ne
            module procedure singleton_ne
        end interface

    contains

        function {F_name_scope}eq(a,b) result (rv)
            use iso_c_binding, only: c_associated
            type({F_derived_name}), intent(IN) ::a,b
            logical :: rv
            if (c_associated(a%{F_derived_member}%addr, b%{F_derived_member}%addr)) then
                rv = .true.
            else
                rv = .false.
            endif
        end function {F_name_scope}eq

        function {F_name_scope}ne(a,b) result (rv)
            use iso_c_binding, only: c_associated
            type({F_derived_name}), intent(IN) ::a,b
            logical :: rv
            if (.not. c_associated(a%{F_derived_member}%addr, b%{F_derived_member}%addr)) then
                rv = .true.
            else
                rv = .false.
            endif
        end function {F_name_scope}ne
 
Generic Interfaces
------------------

Shroud has the ability to create generic interfaces for the routines that are being wrapped.
The generic intefaces groups several functions under a common name.
The compiler will then call the corresponding function based on the argument types used
to call the generic function.


Grouping functions together
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The first case allows multiple C wrapper routines to be called by the same name.
This is done by setting the *F_name_generic* format field.

.. code-block:: yaml

        - decl: void UpdateAsFloat(float arg)
          options:
            F_force_wrapper: True
          format:
            F_name_generic: update_real
        - decl: void UpdateAsDouble(double arg)
          options:
            F_force_wrapper: True
          format:
            F_name_generic: update_real

This allows the correct functions to be called based on the argument type.

.. note:: In this example *F_force_wrapper* is set to *True* since by default
          Shroud will not create explicit wrappers for the functions since only
          native types are used as arguments.
          The generic interface is using ``module procedure``` which requires
          the Fortran wrapper.
          This should be changed in a future version of Shroud.

.. code-block:: fortran

    call update_real(22.0_C_FLOAT)
    call update_real(23.0_C_DOUBLE)

Or more typically as:

.. code-block:: fortran

    call update_real(22.0)
    call update_real(23.0d0)
