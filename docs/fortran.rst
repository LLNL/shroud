.. Copyright (c) 2017-2020, Lawrence Livermore National Security, LLC and
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
        declare
        pre_call
        call
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

Types
-----

The typemap provides several fields used to convert between Fortran and C.

type fields
-----------

.. f_return_type


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



Predefined Types
----------------

Int
^^^

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


Pointers
--------

When a function returns a pointer to a POD type several Fortran
interfaces are possible. When a function returns an ``int *`` the
simplest result is to return a ``type(C_PTR)``.  This is just the raw
pointer returned by C++.  It's also the least useful to the caller
since it cannot be used directly.

If the C++ library function can also provide the length of the
pointer, then its possible to return a Fortran ``POINTER`` or
``ALLOCATABLE`` variable.  This allows the caller to directly use the
returned value of the C++ function.  However, there is a price; the
user will have to release the memory if *owner(caller)* is set.  To
accomplish this with ``POINTER`` arguments, an additional argument is
added to the function which contains information about how to delete
the array.  If the argument is declared Fortran ``ALLOCATABLE``, then
the value of the C++ pointer are copied into a newly allocated Fortran
array. The C++ memory is deleted by the wrapper and it is the callers
responsibility to ``deallocate`` the Fortran array. However, Fortran
will release the array automatically under some conditions when the
caller function returns. If *owner(library)* is set, the Fortran
caller never needs to release the memory.

See :ref:`MemoryManagementAnchor` for details of the implementation.

Functions with ``void *`` arguments are treated differently.  A
``type(C_PTR)`` will be passed by value.  For a ``void **`` argument,
the ``type(C_PTR)`` will be passed by reference (the default).  This
will allow the C wrapper to assign a value to the argument.

.. See clibrary.yaml  passVoidStarStar test

.. code-block:: yaml

    - decl: void passVoidStarStar(void *in+intent(in), void **out+intent(out))

Creates the Fortran interface:

.. code-block:: fortran

        subroutine pass_void_star_star(in, out) &
                bind(C, name="passVoidStarStar")
            use iso_c_binding, only : C_PTR
            implicit none
            type(C_PTR), value, intent(IN) :: in
            type(C_PTR), intent(OUT) :: out
        end subroutine pass_void_star_star

A void pointer may also be used in a C function when any type may be
passed in.  The attribute *assumedtype* can be used to declare a
Fortran argument as assumed-type: ``type(*)``.

.. code-block:: yaml

    - decl: int passAssumedType(void *arg+assumedtype)

.. code-block:: fortran

        function pass_assumed_type(arg) &
                result(SHT_rv) &
                bind(C, name="passAssumedType")
            use iso_c_binding, only : C_INT, C_PTR
            implicit none
            type(*) :: arg
            integer(C_INT) :: SHT_rv
        end function pass_assumed_type


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
 
