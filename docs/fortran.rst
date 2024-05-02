.. Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and
   other Shroud Project Developers.
   See the top-level COPYRIGHT file for details.

   SPDX-License-Identifier: (BSD-3-Clause)


Fortran
=======

This section discusses Fortran specific wrapper details.
This will also include some C wrapper details since some C wrappers
are created specificially to be called by Fortran.

Names
-----

There are several options to mangle the C++ library names into Fortran
names.  By default, names are mangled to convert camel case into snake
case. For example, ``StructAsClass`` into ``struct_as_class``.  Since
Fortran is case insensitive, ``StructAsClass`` and ``structasclass``
are equivalent. By using snake case, the identifier should be easier
for a reader to parse regardless of the case.

The behavior is controlled by the option **F_api_case** which may have
the values *lower*, *upper*, *underscore*, or *preserve*. This option
is used to set the format field **F_name_api** which in turn is used
in several options used to define names consistently:
**F_C_name_template**, **F_name_impl_template**,
**F_name_function_template**, **F_name_generic_template**,
**F_abstract_interface_subprogram_template**,
**F_derived_name_template**, **F_name_typedef_template**.

A Fortran module will be created for the library.  This allows the
compiler to do it's own mangling so it is unnecessary to add an
additional prefix to function names. In contrast, the C wrappers add a
prefix to each wrapper since all names are global.

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
        arg_f_use
        arg_f_decl
        ! splicer begin
        declare      ! local variables
        pre_call
        call  {arg_c_call}
        post_call
        ! splicer end
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

Shroud has the ability to create generic interfaces for the routines
that are being wrapped.  The generic intefaces groups several
functions under a common name.  The compiler will then call the
corresponding function based on the argument types used to call the
generic function.

In several cases generic interfaces are automatically
created. Function overloading and default arguments both create
generic interfaces.

Assumed Rank
^^^^^^^^^^^^

Assumed rank arguments allow a scalar or any rank array to be passed
as an argument. This is added as the attribute *dimension(..)*.  Think
of the ``..`` as a ``:``, used to separate lower and upper bounds,
which fell over. This feature is part of Fortran's *Further
interoperability with C*. First as TS 29113, approved in 2012, then as
part of the Fortran 2018 standard.

.. note:: Shroud does not support *Further Interoperability with C* directly, yet.

Assumed-rank arguments are support by Shroud for older versions of
Fortran by creating a generic interface.  If there are multiple
arguments with assumed-rank, Shroud will give each argument the same
rank for each generic interface.  This handles the common case and
avoids the combinatoral explosion of mixing ranks in a single
function interface.

The ranks used are controlled by the options *F_assumed_rank_min* and
*F_assumed_rank_max* which default to 0, for scalar, and 7.

.. code-block:: yaml

    - decl: int SumValues(const int *values+dimension(..), int nvalues)
      options:
        F_assumed_rank_max: 2

The generated generic interface can be used to pass a scalar, 1d or 2d
array to the C function. In each case ``result`` is 5.

.. code-block:: fortran

    result = sum_array(5, 1)
    result = sum_array([1,1,1,1,1], 5)
      

Grouping Functions Together
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The first case allows multiple C wrapper routines to be called by the
same name.  This is done by setting the *F_name_generic* format field.

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

.. note:: In this example *F_force_wrapper* is set to *True* since by
          default Shroud will not create explicit wrappers for the
          functions since only native types are used as arguments.
          The generic interface is using ``module procedure``` which
          requires the Fortran wrapper.  This should be changed in a
          future version of Shroud.

.. code-block:: fortran

    call update_real(22.0_C_FLOAT)
    call update_real(23.0_C_DOUBLE)

Or more typically as:

.. code-block:: fortran

    call update_real(22.0)
    call update_real(23.0d0)

Argument Coercion
^^^^^^^^^^^^^^^^^

The C compiler will coerce arguments in a function call to the type of
the argument in the prototype.  This makes it very easy to pass an
``float`` to a function which is expecting a ``double``.  Fortran,
which defaults to pass by reference, does not have this feature since
it is passing the address of the argument. This corresponds to C's
behavior since it cannot coerce a ``float *`` to a ``double *``. When
passing a literal ``0.0`` as a ``float`` argument it is necessary to
use ``0.0_C_DOUBLE``.

Shroud can create a generic interface for function which will
coerce arguments similar to C's behavior.
The *fortran_generic* section variations of arguments which will be
used to create a generic interface. For example, when wrapping a function
which takes a ``double``, the ``float`` variation can also be created.

.. code-block:: yaml

    - decl: void GenericReal(double arg)
      fortran_generic:
      - decl: (float arg)
        function_suffix: _float
      - decl: (double arg)
        function_suffix: _double

This will create a generic interface ``generic_real`` with two module
procedures ``generic_real_float`` and ``generic_real_double``.

.. literalinclude:: ../regression/reference/generic/wrapfgeneric.f
   :language: fortran
   :start-after: start generic interface generic_real
   :end-before: end generic interface generic_real
   :dedent: 4

This can be used as

.. code-block:: fortran
                
    call generic_real(0.0)
    call generic_real(0.0d0)
    
    call generic_real_float(0.0)
    call generic_real_double(0.0d0)

When adding *decl* entries to the *fortran_generic* list the original
declaration must also be included, ``double arg`` in this case. When
there are multiple arguments only the arguments which vary need to be
declared.  The other arguments will be the same as the original
*decl* line.

The *function_suffix* line will be used to add a unique string to the
generated Fortran wrappers. Without *function_suffix* each function
will have an integer suffix which is increment for each function.

Scalar and Array Arguments
^^^^^^^^^^^^^^^^^^^^^^^^^^

Shroud can produce a generic interface which allows an argument to be
passed as a scalar or an array. This can help generalize some function
calls where a scalar can be used instead of an array of length one.
This was often used in Fortran code before interfaces were introduced
in Fortran 90. But now when using an interface the compiler will
report an error when passing a scalar where an array is expected.
Likewise, a C function with a pointer argument such as ``int *`` has
no way of knowing how long the array is without being told explicitly.
Thus in C it is easy to pass a pointer to a scalar.

In the *fortran_generic* section one of the declarations can be given
the *rank* attribute which causes the interface to expect an
array. Note that the declaration for the C function does not include
the *rank* attribute.

.. code-block:: yaml

    - decl: int SumArray(int *values, int nvalues)
      fortran_generic:
      - decl: (int *values)
        function_suffix: _scalar
      - decl: (int *values+rank(1))
        function_suffix: _array

The generated generic interface can be used to pass a scalar or array
to the C function.

.. code-block:: fortran

    integer scalar, result
    integer array(5)
    
    scalar = 5
    result = sum_array(scalar, 1)

    array = [1,2,3,4,5]
    result = sum_array(array, 5)

