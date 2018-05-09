.. Copyright (c) 2017-2018, Lawrence Livermore National Security, LLC. 
.. Produced at the Lawrence Livermore National Laboratory 
..
.. LLNL-CODE-738041.
.. All rights reserved. 
..
.. This file is part of Shroud.  For details, see
.. https://github.com/LLNL/shroud. Please also read shroud/LICENSE.
..
.. Redistribution and use in source and binary forms, with or without
.. modification, are permitted provided that the following conditions are
.. met:
..
.. * Redistributions of source code must retain the above copyright
..   notice, this list of conditions and the disclaimer below.
.. 
.. * Redistributions in binary form must reproduce the above copyright
..   notice, this list of conditions and the disclaimer (as noted below)
..   in the documentation and/or other materials provided with the
..   distribution.
..
.. * Neither the name of the LLNS/LLNL nor the names of its contributors
..   may be used to endorse or promote products derived from this
..   software without specific prior written permission.
..
.. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
.. "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
.. LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
.. A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL LAWRENCE
.. LIVERMORE NATIONAL SECURITY, LLC, THE U.S. DEPARTMENT OF ENERGY OR
.. CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
.. EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
.. PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
.. PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
.. LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
.. NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
.. SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
..
.. #######################################################################



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
be controlled directly by the input YAML file::

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
        {F_code}
      end {F_subprogram} {F_name_impl}

    end module {F_module_name}


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
Defaults to *None*.::

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


f_args
^^^^^^

None
Argument in Fortran wrapper to call C.



statements
----------

Statements are used to add additional lines of code for each argument::

      {F_subprogram} {F_name_impl}
        ! arg_f_use
        ! arg_f_decl
        ! pre_call
        {F_code}
        ! post_call
      end {F_subprogram} {F_name_impl}

buf_arg
^^^^^^^


c_local_var
^^^^^^^^^^^

If true, generate a local variable using the C declaration for the argument.
This variable can be used by the pre_call and post_call statements.
A single declaration will be added even if with ``intent(inout)``.

call
^^^^

f_helper
^^^^^^^^

Blank delimited list of helper function names to add to generated Fortran code.
These functions are defined in whelper.py.
There is no current way to add additional functions.


f_module
^^^^^^^^

``USE`` statements to add to Fortran wrapper.
A dictionary of list of ``ONLY`` names::

        f_module=dict(iso_c_binding=['C_SIZE_T']),

declare
^^^^^^^

A list of declarations needed by *pre_call* or *post_call*.
Usually a *c_local_var* is sufficient.


pre_call
^^^^^^^^

Statement to execute before call, often to coerce types when *f_cast* cannot be used.

call
^^^^

Code used to call the function.
Defaults to ``{F_result} = {F_C_call}({F_arg_c_call})``

post_call
^^^^^^^^^

Statement to execute after call.
Can be use to cleanup after *pre_call* or to coerce the return value.

need_wrapper
^^^^^^^^^^^^

If true, the Fortran wrapper will always be created.
This is used when an assignment is needed to do a type coercion;
for example, with logical types.


Predefined Types
----------------

Int
^^^

An ``int`` argument is converted to Fortran with the typemap::

    type: int
    fields:
        f_type: integer(C_INT)
        f_kind: C_INT
        f_module:
            iso_c_binding:
            - C_INT
        f_cast: int({f_var}, C_INT)


Struct Types
------------

A struct in a YAML file creates a ``bind(C)`` derived type for the struct.
A struct may not contain any methods which would cause a v-table to be created.
This will cause an array of structs to be identical in C and C++.

If you want methods on a struct, then use the class keyword.


Class Types
-----------

Fortran can access a C struct created by the C wrapper for each class.
Fortran accesses the {C_capsule_data_type} struct with the ``BIND(C)``
derived type {F_capsule_data_type}.  The C struct is allocated by the
C wrapper and stored as a ``type(C_PTR)`` member.  The Fortran
``POINTER`` {F_derived_member} is associated with this pointer via
``c_f_pointer`` making the the contents directly accessible from
Fortran::

    type, bind(C) :: {F_capsule_data_type}
        type(C_PTR) :: addr = C_NULL_PTR  ! address of C++ memory
        integer(C_INT) :: idtor = 0       ! index of destructor
        integer(C_INT) :: refcount = 0    ! reference count
    end type {F_capsule_data_type}

    type {F_derived_name}
        type(C_PTR), private :: {F_derived_ptr} = C_NULL_PTR
        type({F_capsule_data_type}), pointer :: {F_derived_member} => null()
    contains
        procedure :: {F_name_function} => {F_name_impl}
        generic :: {F_name_generic} => {F_name_function}, ...

        ! F_name_getter, F_name_setter, F_name_instance_get as underscore_name
        procedure :: [F_name_function_template] => [F_name_impl_template]

    end type {F_derived_name}

..        final! :: {F_name_final}

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

Two predicate function are generated to compare derived types::

        interface operator (.eq.)
            module procedure class1_eq
            module procedure singleton_eq
        end interface

        interface operator (.ne.)
            module procedure class1_ne
            module procedure singleton_ne
        end interface

    contains

        function {class_lower}_eq(a,b) result (rv)
            use iso_c_binding, only: c_associated
            type({F_derived_name}), intent(IN) ::a,b
            logical :: rv
            if (c_associated(a%{F_derived_member}%addr, b%{F_derived_member}%addr)) then
                rv = .true.
            else
                rv = .false.
            endif
        end function {class_lower}_eq

        function {class_lower}_ne(a,b) result (rv)
            use iso_c_binding, only: c_associated
            type({F_derived_name}), intent(IN) ::a,b
            logical :: rv
            if (.not. c_associated(a%{F_derived_member}%addr, b%{F_derived_member}%addr)) then
                rv = .true.
            else
                rv = .false.
            endif
        end function {class_lower}_ne
 
