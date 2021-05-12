.. Copyright (c) 2017-2021, Lawrence Livermore National Security, LLC and
   other Shroud Project Developers.
   See the top-level COPYRIGHT file for details.

   SPDX-License-Identifier: (BSD-3-Clause)

.. _DeclarationsAnchor:

Declarations
============

In order for Shroud to create an idiomatic wrapper, it needs to know
how arguments are intended to be used.  This information is supplied
via attributes. This section describes how to describe the arguments
to Shroud in order to implement the desired semantic.

No Arguments
------------

A function with no arguments and which does not return a value, can be
"wrapped" by creating a Fortran interface which allows the function to 
be called directly.

An example is detailed at :ref:`NoReturnNoArguments <example_NoReturnNoArguments>`.

Numeric Arguments
-----------------

Integer and floating point numbers are supported by the
*interoperabilty with C* feature of Fortran 2003.  This includes the
integer types ``short``, ``int``, ``long`` and ``long long``.
Size specific types ``int8_t``, ``int16_t``, ``int32_t``, and
``int64_t`` are also supported.
Floating point types are ``float`` and ``double``.

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

``const int *arg``
    Scalar call-by-reference.
    ``const`` pointers are defaulted to ``+intent(in)``.

``int *arg  +intent(out)``
    If the intent is to return a scalar value from a function,
    add the ``intent(out)`` attribute.
    See example :ref:`PassByReference <example_PassByReference>`.

``const int *arg +rank(1)``
    The ``rank(1)`` attribute will create an assumed-shape
    Fortran dimension for the argument as ``arg(:)``.
    The C library function needs to have some way to determine the length of the
    array.  The length could be assumed by the library function.
    A better option is to add another argument which will explicitly pass
    the length of the array from Fortran - ``int larg+implied(size(arg))``.
    An *implied* argument will not be part of the wrapped API but will still
    be passed to the C++ function.
    See example :ref:`Sum <example_Sum>`.

``int *arg  +intent(out)+deref(allocatable)+dimension(n)``
    Adds the Fortran attribute ``ALLOCATABLE`` to the argument, then
    use the ``ALLOCATE`` statment to allocate memory using *dimension* attribute
    as the shape.
    See example :ref:`truncate_to_int <example_truncate_to_int>`.

``intent **arg +intent(out)``
    Return a pointer in an argument. This is converted into a Fortran
    ``POINTER`` to a scalar.
    See example :ref:`getPtrToScalar <example_getPtrToScalar>`.

``intent **arg +intent(out)+dimension(ncount)``
    Return a pointer in an argument. This is converted into a Fortran
    ``POINTER`` to an array by the *dimension* attribute.
    See example :ref:`getPtrToDynamicArray <example_getPtrToDynamicArray>`.

``intent **arg +intent(out)+deref(raw)``
    Return a pointer in an argument.  The Fortran argument will be
    a ``type(C_PTR)``.  This gives the caller the flexibility to
    dereference the pointer themselves using ``c_f_pointer``.
    This is useful when the shape is not know when the function is called.
    See example :ref:`getRawPtrToFixedArray <example_getRawPtrToFixedArray>`.

``int ***arg +intent(out)``
    Pointers nested to a deeper level are treated as a Fortran ``type(C_PTR)``
    argument.  This gives the user the most flexibility.  The ``type(C_PTR)``
    can be passed back to to library which should know how to cast it.
    There is no checks on the pointer before passing it to the library
    so it's very easy to pass bad values.
    The user can also explicitly dereferences the pointers using ``c_f_pointer``.
    See example :ref:`getRawPtrToInt2d <example_getRawPtrToInt2d>`.

``int **arg +intent(in)``
    Multiple levels of indirection are converted into a ``type(C_PTR)`` argument.
    See below for an exception for ``char **``.
    See example :ref:`checkInt2d <example_checkInt2d>`.
    
``int &min +intent(out)``
    A declaration to a scalar gets converted into pointers in the
    C wrapper.
    See example :ref:`getMinMax <example_getMinMax>`.

``int *&arg``
   Return a pointer in an argument.  From Fortran, this is the same
   as ``int **arg``.  See above examples.


Numeric Functions
-----------------

``int *func()``
    Return a Fortran ``POINTER`` to a scalar.
    See example :ref:`returnIntPtrToScalar <example_returnIntPtrToScalar>`.

``int *func() +dimension(10)``
    Return a Fortran ``POINTER`` to a array.
    See example :ref:`returnIntPtrToFixedArray <example_returnIntPtrToFixedArray>`.

``int *func() +deref(scalar)``
    Return a scalar.
    See example :ref:`returnIntScalar <example_returnIntScalar>`.


Bool
----

C and C++ functions with a ``bool`` argument generate a Fortran wrapper with
a ``logical`` argument.  One of the goals of Shroud is to produce an
idiomatic interface.  Converting the types in the wrapper avoids the
awkwardness of requiring the Fortran user to passing in
``.true._c_bool`` instead of just ``.true.``.
Using an integer for a ``bool`` argument is not portable since
some compilers use 1 for ``.true.`` and others use -1.


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

..
 This is the C++ prototype with the addition of **+len(30)**.  This
 attribute defines the declared length of the returned string.  Since
 *Function4a* is returning a ``std::string`` the contents of the string
 must be copied out into a Fortran variable so that the ``std::string``
 may be deallocated by C++. Otherwise, it would leak memory.

 The downside of this approach is that the maximum length of the return argument must be 
 known in advance.  By leaving off the **+len(30)**, Shroud will create an ``ALLOCATABLE``
 function which will allocate a ``CHARACTER`` variable of the correct length:


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

``char **names +intent(in)``
    This is a standard C idiom for an array of ``NULL`` terminated strings.
    Shroud takes an array of ``CHARACTER(len=*) arg(:)`` and creates the
    C data structure by copying the data and adding the terminating ``NULL``.
    See example :ref:`acceptCharArrayIn <example_acceptCharArrayIn>`.

.. XXX 

std::string
-----------

..
 The C wrapper uses ``char *`` for ``std::string`` arguments which
 Fortran declares as ``character``.
 The argument is passed to the ``std::string`` constructor.
 In addition the length of the data in each string is computed using ``len_trim``
 and passed down.
 No trailing ``NULL`` is required.
 This avoids copying the string in Fortran which would be necessary to
 append the trailing ``C_NULL_CHAR``.
 The return value is added as another argument along with its declared length
 computed using ``len``:

 The contents of the ``std::string`` are copied into the result argument and blank
 filled by ``ShroudStrCopy``.
 Before the C wrapper returns, ``SHT_rv`` will be deleted by the compiler.



``std::string & arg``
    ``arg`` will default to ``intent(inout)``.
    See example :ref:`acceptStringReference <example_acceptStringReference>`.


char functions
--------------

Functions which return a ``char *`` provide an additional challenge.
Taken literally they should return a ``type(C_PTR)``.  And if you call
the C library function via the interface, that's what you get.  However,
Shroud provides several options to provide a more idiomatic usage.

Each of these declaration call identical C++ functions but they are
wrapped differently.

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
    Create a Fortran subroutine with an additional ``CHARACTER``
    argument for the C function result. Any size character string can
    be returned limited by the size of the Fortran argument.  The
    argument is defined by the *F_string_result_as_arg* format string.
    Works with Fortran 90.
    See example :ref:`getCharPtr3 <example_getCharPtr3>`.


.. XXX returning a scalar char will pass the result to the C wrapper
   as an ``char *`` argument.  pgi and cray compilers had issues with
   bind(C) functions which returned CHARACTER(len=1,kind=C_CHAR)
   valgrind reported uninitialized variables when calling the Fortran
   wrapper.  i.e.  CHARACTER is not considered a scalar type.

string functions
----------------

Functions which return ``std::string`` values are similar but must provide the
extra step of converting the result into a ``char *``.

``const string &``
    See example :ref:`getConstStringRefPure <example_getConstStringRefPure>`.

std::vector
-----------

A ``std::vector`` argument for a C++ function can be created from a
Fortran array.  The address and size of the array is extracted and
passed to the C wrapper to create the ``std::vector``


``const std::vector<int> &arg``
    ``arg`` defaults to ``intent(in)`` since it is const.
    See example :ref:`vector_sum <example_vector_sum>`.

``std::vector<int> &arg``
    See example :ref:`vector_iota_out <example_vector_iota_out>`.

See example :ref:`vector_iota_out_alloc <example_vector_iota_out_alloc>`.

See example :ref:`vector_iota_inout_alloc <example_vector_iota_inout_alloc>`.

On ``intent(in)``, the ``std::vector`` constructor copies the values
from the input pointer.  With ``intent(out)``, the values are copied
after calling the function.

.. note:: With ``intent(out)``, if *vector_iota* changes the size of
          ``arg`` to be longer than the original size of the Fortran
          argument, the additional values will not be copied. 

Void Pointers
-------------

The Fortran 2003 stardard added the ``type(C_PTR)`` derived type 
which is used to hold a C ``void *``.
Fortran is not able to directly dereference ``type(C_PTR)`` variables.
The function ``c_f_pointer`` must be used.

``void *arg``
    If the intent is to be able to pass any variable to the function,
    add the ``+assumedtype`` attribute.
    ``type(*)`` is only available with TS 29113.
    The Fortran wrapper will only accept scalar arguments.
    To pass an array, add the ``dimension`` attribute
    See examples :ref:`passAssumedType <example_passAssumedType>` and
    :ref:`passAssumedTypeDim <example_passAssumedTypeDim>`.

``void *arg``
    Passes the value of a ``type(C_PTR)`` argument.
    See example :ref:`passVoidStarStar <example_passVoidStarStar>`.

``void **arg``
    Used to return a ``void *`` from a function in an argument.
    Passes the address of a ``type(C_PTR)`` argument.
    See example :ref:`passVoidStarStar <example_passVoidStarStar>`.

.. _DeclAnchor_Function_Pointers:

Function Pointers
-----------------

C or C++ arguments which are pointers to functions are supported.
The function pointer type is wrapped using a Fortran ``abstract interface``.
Only C compatible arguments in the function pointer are supported since
no wrapper for the function pointer is created.  It must be callable 
directly from Fortran.

``int (*incr)(int)``
    Create a Fortran abstract interface for the function pointer.
    Only functions which match the interface can be used as a dummy argument.
    See example :ref:`callback1 <example_callback1>`.

``void (*incr)()``
    Adding the ``external`` attribute will allow any function to be passed.
    In C this is accomplished by using a cast.
    See example :ref:`callback1c <example_callback1c>`.

The ``abstract interface`` is named from option
**F_abstract_interface_subprogram_template** which defaults to
``{underscore_name}_{argname}`` where *argname* is the name of the
function argument.

If the function pointer uses an abstract declarator
(no argument name), the argument name is created from option
**F_abstract_interface_argument_template** which defaults to
``arg{index}`` where *index* is the 0-based argument index.
When a name is given to a function pointer argument,
it is always used in the ``abstract interface``.

To change the name of the subprogram or argument, change the option.
There are no format fields **F_abstract_interface_subprogram** or
**F_abstract_interface_argument** since they vary by argument (or
argument to an argument):

.. code-block:: yaml

    options:
      F_abstract_interface_subprogram_template: custom_funptr
      F_abstract_interface_argument_template: XX{index}arg

It is also possible to pass a function which will accept any function
interface as the dummy argument. This is done by adding the *external*
attribute.  A Fortran wrapper function is created with an ``external``
declaration for the argument. The C function is called via an interace
with the ``bind(C)`` attribute.  In the interface, an ``abstract
interface`` for the function pointer argument is used.  The user's
library is responsible for calling the argument correctly since the
interface is not preserved by the ``external`` declaration.

Struct
------

See example :ref:`passStruct1 <example_passStruct1>`.

See example :ref:`passStructByValue <example_passStructByValue>`.
