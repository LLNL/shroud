.. Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and
   other Shroud Project Developers.
   See the top-level COPYRIGHT file for details.

   SPDX-License-Identifier: (BSD-3-Clause)

.. All of the examples are ordered as
   C or C++ code from user's library
   YAML input
   C++ wrapper
   Fortran interface
   Fortran wrapper
   Fortran example usage
   C++ example usage

Sample Fortran Wrappers
=======================

This chapter gives details of the generated code.
It's intended for users who want to understand the details
of how the wrappers are created.

All of these examples are derived from tests in the ``regression``
directory.

.. _example_NoReturnNoArguments:

No Arguments
------------

C library function in :file:`clibrary.c`:

.. literalinclude:: ../regression/run/clibrary/clibrary.c
   :language: c
   :start-after: start NoReturnNoArguments
   :end-before: end NoReturnNoArguments

:file:`clibrary.yaml`:

.. code-block:: yaml

    - decl: void NoReturnNoArguments()

Fortran calls C via the following interface:

.. literalinclude:: ../regression/reference/clibrary/wrapfclibrary.f
   :language: fortran
   :start-after: start no_return_no_arguments
   :end-before: end no_return_no_arguments
   :dedent: 4

If wrapping a C++ library, a function with a C API will be created
that Fortran can call.

.. literalinclude:: ../regression/reference/tutorial/wrapTutorial.cpp
   :language: c
   :start-after: start TUT_NoReturnNoArguments
   :end-before: end TUT_NoReturnNoArguments

Fortran usage:

.. code-block:: fortran

    use tutorial_mod
    call no_return_no_arguments

The C++ usage is similar:

.. code-block:: c++

    #include "tutorial.hpp"

    tutorial::NoReturnNoArguments();


Numeric Types
-------------

.. ############################################################

.. _example_PassByValue:

PassByValue
^^^^^^^^^^^

C library function in :file:`clibrary.c`:

.. literalinclude:: ../regression/run/clibrary/clibrary.c
   :language: c
   :start-after: start PassByValue
   :end-before: end PassByValue

:file:`clibrary.yaml`:

.. code-block:: yaml

    - decl: double PassByValue(double arg1, int arg2)

Both types are supported directly by the ``iso_c_binding`` module
so there is no need for a Fortran function.
The C function can be called directly by the Fortran interface
using the ``bind(C)`` keyword.

Fortran calls C via the following interface:

.. literalinclude:: ../regression/reference/clibrary/wrapfclibrary.f
   :language: fortran
   :start-after: start pass_by_value
   :end-before: end pass_by_value
   :dedent: 4

Fortran usage:

.. code-block:: fortran

    real(C_DOUBLE) :: rv_double
    rv_double = pass_by_value(1.d0, 4)
    call assert_true(rv_double == 5.d0)

.. ############################################################

.. _example_PassByReference:

PassByReference
^^^^^^^^^^^^^^^

C library function in :file:`clibrary.c`:

.. literalinclude:: ../regression/run/clibrary/clibrary.c
   :language: c
   :start-after: start PassByReference
   :end-before: end PassByReference

:file:`clibrary.yaml`:

.. code-block:: yaml

    - decl: void PassByReference(double *arg1+intent(in), int *arg2+intent(out))

Fortran calls C via the following interface:

.. literalinclude:: ../regression/reference/clibrary/wrapfclibrary.f
   :language: fortran
   :start-after: start pass_by_reference
   :end-before: end pass_by_reference
   :dedent: 4

Example usage:

.. code-block:: fortran

    integer(C_INT) var
    call pass_by_reference(3.14d0, var)
    call assert_equals(3, var)

.. ############################################################

.. _example_Sum:

Sum
^^^

C++ library function from :file:`pointers.cpp`:

.. literalinclude:: ../regression/run/pointers/pointers.c
   :language: c
   :start-after: start Sum
   :end-before: end Sum

:file:`pointers.yaml`:

.. code-block:: yaml

   - decl: void Sum(int len +implied(size(values)),
                    int *values +rank(1)+intent(in),
                    int *result +intent(out))

The ``POI`` prefix to the function names is derived from 
the format field *C_prefix* which defaults to the first three letters
of the *library* field, in this case *pointers*.
This is a C++ file which provides a C API via ``extern "C"``.

:file:`wrappointers.cpp`:

.. literalinclude:: ../regression/reference/pointers-cxx/wrappointers.cpp
   :language: c
   :start-after: start POI_sum
   :end-before: end POI_sum

Fortran calls C via the following interface:

.. literalinclude:: ../regression/reference/pointers-cxx/wrapfpointers.f
   :language: fortran
   :start-after: start c_sum
   :end-before: end c_sum
   :dedent: 4

The Fortran wrapper:

.. literalinclude:: ../regression/reference/pointers-cxx/wrapfpointers.f
   :language: fortran
   :start-after: start sum
   :end-before: end sum
   :dedent: 4

Example usage:

.. code-block:: fortran

    integer(C_INT) rv_int
    call sum([1,2,3,4,5], rv_int)
    call assert_true(rv_int .eq. 15, "sum")

.. ############################################################

.. _example_truncate_to_int:

truncate_to_int
^^^^^^^^^^^^^^^
Sometimes it is more convenient to have the wrapper allocate an
``intent(out)`` array before passing it to the C++ function.  This can
be accomplished by adding the *deref(allocatable)* attribute.

C++ library function from :file:`pointers.c`:

.. literalinclude:: ../regression/run/pointers/pointers.c
   :language: c
   :start-after: start truncate_to_int
   :end-before: end truncate_to_int

:file:`pointers.yaml`:

.. code-block:: yaml

    - decl: void truncate_to_int(double * in     +intent(in)  +rank(1),
                                 int *    out    +intent(out)
                                                 +deref(allocatable)+dimension(size(in)),
                                 int      sizein +implied(size(in)))
      

Fortran calls C via the following interface:

.. literalinclude:: ../regression/reference/pointers-c/wrapfpointers.f
   :language: fortran
   :start-after: start c_truncate_to_int
   :end-before: end c_truncate_to_int
   :dedent: 4

The Fortran wrapper:

.. literalinclude:: ../regression/reference/pointers-c/wrapfpointers.f
   :language: fortran
   :start-after: start truncate_to_int
   :end-before: end truncate_to_int
   :dedent: 4

Example usage:

.. code-block:: fortran

    integer(c_int), allocatable :: out_int(:)
    call truncate_to_int([1.2d0, 2.3d0, 3.4d0, 4.5d0], out_int)

Numeric Pointers
----------------

.. ############################################################

.. _example_getRawPtrToFixedArray:

getRawPtrToFixedArray
^^^^^^^^^^^^^^^^^^^^^

C++ library function from :file:`pointers.c`:

.. literalinclude:: ../regression/run/pointers/pointers.c
   :language: c
   :start-after: start getRawPtrToFixedArray
   :end-before: end getRawPtrToFixedArray

:file:`pointers.yaml`:

.. code-block:: yaml

    - decl: void getRawPtrToFixedArray(int **count+intent(out)+deref(raw))

Fortran calls C via the following interface:

.. literalinclude:: ../regression/reference/pointers-c/wrapfpointers.f
   :language: fortran
   :start-after: start get_raw_ptr_to_fixed_array
   :end-before: end get_raw_ptr_to_fixed_array
   :dedent: 4

Example usage:

.. code-block:: fortran

    type(C_PTR) :: cptr_array
    call get_raw_ptr_to_fixed_array(cptr_array)

    
.. ############################################################

.. _example_getPtrToScalar:

getPtrToScalar
^^^^^^^^^^^^^^

C++ library function from :file:`pointers.c`:

.. literalinclude:: ../regression/run/pointers/pointers.c
   :language: c
   :start-after: start getPtrToScalar
   :end-before: end getPtrToScalar

:file:`pointers.yaml`:

.. code-block:: yaml

    - decl: void getPtrToScalar(int **nitems+intent(out))

This is a C file which provides the bufferify function.

:file:`wrappointers.c`:

.. literalinclude:: ../regression/reference/pointers-c/wrappointers.c
   :language: c
   :start-after: start POI_getPtrToScalar_bufferify
   :end-before: end POI_getPtrToScalar_bufferify

Fortran calls C via the following interface:

.. literalinclude:: ../regression/reference/pointers-c/wrapfpointers.f
   :language: fortran
   :start-after: start c_get_ptr_to_scalar
   :end-before: end c_get_ptr_to_scalar
   :dedent: 4

The Fortran wrapper:

.. literalinclude:: ../regression/reference/pointers-c/wrapfpointers.f
   :language: fortran
   :start-after: start get_ptr_to_scalar
   :end-before: end get_ptr_to_scalar
   :dedent: 4

Assigning to ``iscalar`` will modify the C++ variable.
Example usage:

.. code-block:: fortran

    integer(C_INT), pointer :: iscalar
    call get_ptr_to_scalar(iscalar)
    iscalar = 0

.. ############################################################

.. _example_getPtrToDynamicArray:

getPtrToDynamicArray
^^^^^^^^^^^^^^^^^^^^

C++ library function from :file:`pointers.c`:

.. literalinclude:: ../regression/run/pointers/pointers.c
   :language: c
   :start-after: start getPtrToDynamicArray
   :end-before: end getPtrToDynamicArray

:file:`pointers.yaml`:

.. code-block:: yaml

    - decl: void getPtrToDynamicArray(int **count+intent(out)+dimension(ncount),
                                      int *ncount+intent(out)+hidden)

This is a C file which provides the bufferify function.

:file:`wrappointers.c`:

.. literalinclude:: ../regression/reference/pointers-c/wrappointers.c
   :language: c
   :start-after: start POI_getPtrToDynamicArray_bufferify
   :end-before: end POI_getPtrToDynamicArray_bufferify

Fortran calls C via the following interface:

.. literalinclude:: ../regression/reference/pointers-c/wrapfpointers.f
   :language: fortran
   :start-after: start c_get_ptr_to_dynamic_array
   :end-before: end c_get_ptr_to_dynamic_array
   :dedent: 4

The Fortran wrapper:

.. literalinclude:: ../regression/reference/pointers-c/wrapfpointers.f
   :language: fortran
   :start-after: start get_ptr_to_dynamic_array
   :end-before: end get_ptr_to_dynamic_array
   :dedent: 4

Assigning to ``iarray`` will modify the C++ variable.
Example usage:

.. code-block:: fortran

    integer(C_INT), pointer :: iarray(:)
    call get_ptr_to_dynamic_array(iarray)
    iarray = 0

.. ############################################################

.. _example_getRawPtrToInt2d:

getRawPtrToInt2d
^^^^^^^^^^^^^^^^

`global_int2d` is a two dimensional array of non-contiguous rows.
C stores the address of each row.
Shroud can only deal with this as a ``type(C_PTR)`` and expects the
user to dereference the address.

C++ library function from :file:`pointers.c`:

.. code-block:: yaml

    static int global_int2d_1[] = {1,2,3};
    static int global_int2d_2[] = {4,5};
    static int *global_int2d[] = {global_int2d_1, global_int2d_2};

    void getRawPtrToInt2d(int ***arg)
    {
        *arg = (int **) global_int2d;
    }

:file:`pointers.yaml`:

.. code-block:: yaml

    - decl: void getRawPtrToInt2d(int ***arg +intent(out))

Fortran calls C via the following interface:

.. literalinclude:: ../regression/reference/pointers-c/wrapfpointers.f
   :language: fortran
   :start-after: start get_raw_ptr_to_int2d
   :end-before: end get_raw_ptr_to_int2d
   :dedent: 4

Example usage:

.. code-block:: fortran

    type(C_PTR) :: addr
    type(C_PTR), pointer :: array2d(:)
    integer(C_INT), pointer :: row1(:), row2(:)
    integer total

    call get_raw_ptr_to_int2d(addr)

    ! Dereference the pointers into two 1d arrays.
    call c_f_pointer(addr, array2d, [2])
    call c_f_pointer(array2d(1), row1, [3])
    call c_f_pointer(array2d(2), row2, [2])

    total = row1(1) + row1(2) + row1(3) + row2(1) + row2(2)
    call assert_equals(15, total)

.. ############################################################

.. _example_checkInt2d:

checkInt2d
^^^^^^^^^^

Example of using the ``type(C_PTR)`` returned 
:ref:`getRawPtrToInt2d <example_getRawPtrToInt2d>`.

:file:`pointers.yaml`:

.. code-block:: yaml

    - decl: int checkInt2d(int **arg +intent(in))

Fortran calls C via the following interface.
Note the use of ``VALUE`` attribute.

.. literalinclude:: ../regression/reference/pointers-c/wrapfpointers.f
   :language: fortran
   :start-after: start check_int2d
   :end-before: end check_int2d
   :dedent: 4

Example usage:

.. code-block:: fortran

    type(C_PTR) :: addr
    integer total

    call get_raw_ptr_to_int2d(addr)
    total = check_int2d(addr)
    call assert_equals(15, total)


.. ############################################################

.. _example_getMinMax:

getMinMax
^^^^^^^^^

No Fortran function is created.  Only an interface to a C wrapper
which dereference the pointers so they can be treated as references.

C++ library function in :file:`tutorial.cpp`:

.. literalinclude:: ../regression/run/tutorial/tutorial.cpp
   :language: c
   :start-after: start getMinMax
   :end-before: end getMinMax

:file:`tutorial.yaml`:

.. code-block:: yaml

    - decl: void getMinMax(int &min +intent(out), int &max +intent(out))

The C wrapper:

.. literalinclude:: ../regression/reference/tutorial/wrapTutorial.cpp
   :language: c
   :start-after: start TUT_getMinMax
   :end-before: end TUT_getMinMax

Fortran calls C via the following interface:

.. literalinclude:: ../regression/reference/tutorial/wrapftutorial.f
   :language: fortran
   :start-after: start get_min_max
   :end-before: end get_min_max
   :dedent: 4

Fortran usage:

.. code-block:: fortran

    call get_min_max(minout, maxout)
    call assert_equals(-1, minout, "get_min_max minout")
    call assert_equals(100, maxout, "get_min_max maxout")

.. ############################################################

.. _example_returnIntPtrToScalar:

returnIntPtrToScalar
^^^^^^^^^^^^^^^^^^^^

.. fc_statememnt f_native_*_result_pointer

:file:`pointers.yaml`:

.. code-block:: yaml

    - decl: int *returnIntPtrToScalar(void)

Fortran calls C via the following interface:

.. literalinclude:: ../regression/reference/pointers-c/wrapfpointers.f
   :language: fortran
   :start-after: start c_return_int_ptr_to_scalar
   :end-before: end c_return_int_ptr_to_scalar
   :dedent: 4

The Fortran wrapper:

.. literalinclude:: ../regression/reference/pointers-c/wrapfpointers.f
   :language: fortran
   :start-after: start return_int_ptr_to_scalar
   :end-before: end return_int_ptr_to_scalar
   :dedent: 4

Example usage:

.. code-block:: fortran

    integer(C_INT), pointer :: irvscalar
    irvscalar => return_int_ptr_to_scalar()

.. ############################################################

.. _example_returnIntPtrToFixedArray:

returnIntPtrToFixedArray
^^^^^^^^^^^^^^^^^^^^^^^^

:file:`pointers.yaml`:

.. code-block:: yaml

    - decl: int *returnIntPtrToFixedArray(void) +dimension(10)

This is a C file which provides the bufferify function.

:file:`wrappointers.c`:

.. literalinclude:: ../regression/reference/pointers-c/wrappointers.c
   :language: c
   :start-after: start POI_returnIntPtrToFixedArray_bufferify
   :end-before: end POI_returnIntPtrToFixedArray_bufferify

Fortran calls C via the following interface:

.. literalinclude:: ../regression/reference/pointers-c/wrapfpointers.f
   :language: fortran
   :start-after: start c_return_int_ptr_to_fixed_array_bufferify
   :end-before: end c_return_int_ptr_to_fixed_array_bufferify
   :dedent: 4

The Fortran wrapper:

.. literalinclude:: ../regression/reference/pointers-c/wrapfpointers.f
   :language: fortran
   :start-after: start return_int_ptr_to_fixed_array
   :end-before: end return_int_ptr_to_fixed_array
   :dedent: 4

Example usage:

.. code-block:: fortran

    integer(C_INT), pointer :: irvarray(:)
    irvarray => return_int_ptr_to_fixed_array()

.. ############################################################

.. _example_returnIntScalar:

returnIntScalar
^^^^^^^^^^^^^^^

:file:`pointers.yaml`:

.. code-block:: yaml

    - decl: int *returnIntScalar(void) +deref(scalar)

This is a C file which provides the bufferify function.

:file:`wrappointers.c`:

.. literalinclude:: ../regression/reference/pointers-c/wrappointers.c
   :language: c
   :start-after: start POI_returnIntScalar
   :end-before: end POI_returnIntScalar

Fortran calls C via the following interface:

.. literalinclude:: ../regression/reference/pointers-c/wrapfpointers.f
   :language: fortran
   :start-after: start return_int_scalar
   :end-before: end return_int_scalar
   :dedent: 4

Example usage:

.. code-block:: fortran

    integer :: ivalue
    ivalue = return_int_scalar()

.. ############################################################

.. _example_ReturnIntPtrDimPointer:

returnIntPtrDimPointer
^^^^^^^^^^^^^^^^^^^^^^

Return a Fortran pointer to an array.
The length of the array is returned from C++ in the *len* argument.
This argument sets the *hidden* attribute since it is not needed in
the Fortran wrapper. It will be used in the ``c_f_pointer`` call to
set the length of the array.

..    - decl: int *ReturnIntPtrDimPointer(int *len+intent(out)+hidden) +deref(pointer)

The input is in file :file:`ownership.yaml`.

.. literalinclude:: ../regression/input/ownership.yaml
   :language: yaml
   :start-after: start ReturnIntPtrDimPointer
   :end-before: end ReturnIntPtrDimPointer

The C wrapper calls the C++ function from an ``extern C`` wrapper.
In does not hide the *len* argument.
This function does not use the *deref* attribute.
      
.. literalinclude:: ../regression/reference/ownership/wrapownership.cpp
   :language: c++
   :start-after: start OWN_ReturnIntPtrDimPointer
   :end-before: end OWN_ReturnIntPtrDimPointer

The bufferify function passes an argument to contain the meta data of the array.
It is written to :file:`wrapownership.cpp`.

.. literalinclude:: ../regression/reference/ownership/wrapownership.cpp
   :language: c++
   :start-after: start OWN_ReturnIntPtrDimPointer_bufferify
   :end-before: end OWN_ReturnIntPtrDimPointer_bufferify

Fortran calls the bufferify function in :file:`wrapfownership.f`.

.. literalinclude:: ../regression/reference/ownership/wrapfownership.f
   :language: fortran
   :start-after: start return_int_ptr_dim_pointer
   :end-before: end return_int_ptr_dim_pointer
   :dedent: 4

Fortran usage:

.. code-block:: fortran

    integer(C_INT), pointer :: ivalue(:)
    integer len

    ivalue => return_int_ptr_dim_pointer()
    len = size(ivalue)

.. ############################################################

.. _example_ReturnIntPtrDimAlloc:

returnIntPtrDimAlloc
^^^^^^^^^^^^^^^^^^^^

Convert a pointer returned from C++ into a Fortran allocatable array.
To do this, memory is allocated in Fortran then the C++ values are copied
into it.
The advantage is that the user does not have to worry about releasing the
C++ memory.
The length of the array is returned from C++ in the *len* argument.
This argument sets the *hidden* attribute since it is not needed in
the Fortran wrapper.

..    - decl: int *ReturnIntPtrDimAlloc(int *len+intent(out)+hidden) +deref(allocatable)

The input is in file :file:`ownership.yaml`.

.. literalinclude:: ../regression/input/ownership.yaml
   :language: yaml
   :start-after: start ReturnIntPtrDimAlloc
   :end-before: end ReturnIntPtrDimAlloc

The C wrapper calls the C++ function from an ``extern C`` wrapper.
In does not hide the *len* argument.
This function does not use the *deref* attribute.
      
.. literalinclude:: ../regression/reference/ownership/wrapownership.cpp
   :language: c++
   :start-after: start OWN_ReturnIntPtrDimAlloc
   :end-before: end OWN_ReturnIntPtrDimAlloc

The bufferify function passes an argument to contain the meta data of the array.
It is written to :file:`wrapownership.cpp`.

.. literalinclude:: ../regression/reference/ownership/wrapownership.cpp
   :language: c++
   :start-after: start OWN_ReturnIntPtrDimAlloc_bufferify
   :end-before: end OWN_ReturnIntPtrDimAlloc_bufferify

Fortran calls the bufferify function in :file:`wrapfownership.f`.

.. literalinclude:: ../regression/reference/ownership/wrapfownership.f
   :language: fortran
   :start-after: start return_int_ptr_dim_alloc
   :end-before: end return_int_ptr_dim_alloc
   :dedent: 4

Fortran usage:

.. code-block:: fortran

    integer(C_INT), allocatable :: ivalue(:)
    integer len

    ivalue = return_int_ptr_dim_alloc()
    len = size(ivalue)

Bool
----

.. ############################################################

.. _example_checkBool:

checkBool
^^^^^^^^^

Assignments are done in the Fortran wrapper to convert between
``logical`` and ``logical(C_BOOL)``.

C library function in :file:`clibrary`:

.. literalinclude:: ../regression/run/clibrary/clibrary.c
   :language: c
   :start-after: start checkBool
   :end-before: end checkBool

:file:`clibrary.yaml`:

.. code-block:: yaml

    - decl: void checkBool(const bool arg1,
                           bool *arg2+intent(out),
                           bool *arg3+intent(inout))

Fortran calls C via the following interface:

.. literalinclude:: ../regression/reference/clibrary/wrapfclibrary.f
   :language: fortran
   :start-after: start c_check_bool
   :end-before: end c_check_bool
   :dedent: 4

The Fortran wrapper:

.. literalinclude:: ../regression/reference/clibrary/wrapfclibrary.f
   :language: fortran
   :start-after: start check_bool
   :end-before: end check_bool
   :dedent: 4

Fortran usage:

.. code-block:: fortran

    logical rv_logical, wrk_logical
    rv_logical = .true.
    wrk_logical = .true.
    call check_bool(.true., rv_logical, wrk_logical)
    call assert_false(rv_logical)
    call assert_false(wrk_logical)


Character
---------

.. ############################################################

.. _example_acceptName:

acceptName
^^^^^^^^^^

Pass a ``NULL`` terminated string to a C function.
The string will be unchanged.

C library function in :file:`clibrary.c`:

.. literalinclude:: ../regression/run/clibrary/clibrary.c
   :language: c
   :start-after: start acceptName
   :end-before: end acceptName

:file:`clibrary.yaml`:

.. code-block:: yaml

  - decl: void acceptName(const char *name)

Fortran calls C via the following interface:

.. literalinclude:: ../regression/reference/clibrary/wrapfclibrary.f
   :language: fortran
   :start-after: start c_accept_name
   :end-before: end c_accept_name
   :dedent: 4

The Fortran wrapper:

.. literalinclude:: ../regression/reference/clibrary/wrapfclibrary.f
   :language: fortran
   :start-after: start accept_name
   :end-before: end accept_name
   :dedent: 4

No C wrapper is required since the Fortran wrapper creates a NULL
terminated string by calling the Fortran intrinsic function ``trim``
and concatenating ``C_NULL_CHAR`` (from ``iso_c_binding``).  This can
be done since the argument ``name`` is ``const`` which sets the
attribute *intent(in)*.

Fortran usage:

.. code-block:: fortran

    call accept_name("spot")

.. ############################################################

.. _example_returnOneName:

returnOneName
^^^^^^^^^^^^^

Pass the pointer to a buffer which the C library will fill.  The
length of the string is implicitly known by the library to not exceed
the library variable ``MAXNAME``.

C library function in :file:`clibrary.c`:

.. literalinclude:: ../regression/run/clibrary/clibrary.c
   :language: c
   :start-after: start returnOneName
   :end-before: end returnOneName

:file:`clibrary.yaml`:

.. code-block:: yaml

    - decl: void returnOneName(char *name1+intent(out)+charlen(MAXNAME))

The C wrapper:

.. literalinclude:: ../regression/reference/clibrary/wrapClibrary.c
   :language: c
   :start-after: start CLI_returnOneName_bufferify
   :end-before: end CLI_returnOneName_bufferify

Fortran calls C via the following interface:

.. literalinclude:: ../regression/reference/clibrary/wrapfclibrary.f
   :language: fortran
   :start-after: start c_return_one_name_bufferify
   :end-before: end c_return_one_name_bufferify
   :dedent: 4

The Fortran wrapper:

.. literalinclude:: ../regression/reference/clibrary/wrapfclibrary.f
   :language: fortran
   :start-after: start return_one_name
   :end-before: end return_one_name
   :dedent: 4

Fortran usage:

.. code-block:: fortran

    name1 = " "
    call return_one_name(name1)
    call assert_equals("bill", name1)

.. ############################################################

.. _example_passCharPtr:

passCharPtr
^^^^^^^^^^^

The function ``passCharPtr(dest, src)`` is equivalent to the Fortran
statement ``dest = src``:

C++ library function in :file:`strings.cpp`:

.. literalinclude:: ../regression/run/strings/strings.cpp
   :language: c
   :start-after: start passCharPtr
   :end-before: end passCharPtr

:file:`strings.yaml`:

.. code-block:: yaml

    - decl: void passCharPtr(char * dest+intent(out)+charlen(40),
                             const char *src)

The intent of ``dest`` must be explicit.  It defaults to *intent(inout)*
since it is a pointer.
``src`` is implied to be *intent(in)* since it is ``const``.
This single line will create five different wrappers.

The native C version.
The only feature this provides to Fortran is the ability
to call a C++ function by wrapping it in an ``extern "C"`` function.
The user is responsible for providing the ``NULL`` termination.
The result in ``str`` will also be ``NULL`` terminated instead of 
blank filled.:

.. literalinclude:: ../regression/reference/strings/wrapstrings.cpp
   :language: c
   :start-after: start STR_passCharPtr
   :end-before: end STR_passCharPtr

The C wrapper:

.. literalinclude:: ../regression/reference/strings/wrapstrings.cpp
   :language: c
   :start-after: start STR_passCharPtr_bufferify
   :end-before: end STR_passCharPtr_bufferify

Fortran calls C via the following interface:

.. literalinclude:: ../regression/reference/strings/wrapfstrings.f
   :language: fortran
   :start-after: start c_pass_char_ptr
   :end-before: end c_pass_char_ptr
   :dedent: 4

.. literalinclude:: ../regression/reference/strings/wrapfstrings.f
   :language: fortran
   :start-after: start c_pass_char_ptr_bufferify
   :end-before: end c_pass_char_ptr_bufferify
   :dedent: 4

The Fortran wrapper:

.. literalinclude:: ../regression/reference/strings/wrapfstrings.f
   :language: fortran
   :start-after: start pass_char_ptr
   :end-before: end pass_char_ptr
   :dedent: 4

The function can be called without the user aware that it is written in C++:

.. code-block:: fortran

    character(30) str
    call pass_char_ptr(dest=str, src="mouse")


.. ############################################################

.. _example_ImpliedTextLen:

ImpliedTextLen
^^^^^^^^^^^^^^

Pass the pointer to a buffer which the C library will fill.  The
length of the buffer is passed in ``ltext``.  Since Fortran knows the
length of ``CHARACTER`` variable, the Fortran wrapper does not need to
be explicitly told the length of the variable.  Instead its value can
be defined with the *implied* attribute.

This can be used to emulate the behavior of most Fortran compilers
which will pass an additional, hidden argument which contains the
length of a ``CHARACTER`` argument.

C library function in :file:`clibrary.c`:

.. literalinclude:: ../regression/run/clibrary/clibrary.c
   :language: c
   :start-after: start ImpliedTextLen
   :end-before: end ImpliedTextLen

:file:`clibrary.yaml`:

.. code-block:: yaml

    - decl: void ImpliedTextLen(char *text+intent(out)+charlen(MAXNAME),
                                int ltext+implied(len(text)))

The C wrapper:

.. literalinclude:: ../regression/reference/clibrary/wrapClibrary.c
   :language: c
   :start-after: start CLI_ImpliedTextLen_bufferify
   :end-before: end CLI_ImpliedTextLen_bufferify

Fortran calls C via the following interface:

.. literalinclude:: ../regression/reference/clibrary/wrapfclibrary.f
   :language: fortran
   :start-after: start c_implied_text_len_bufferify
   :end-before: end c_implied_text_len_bufferify
   :dedent: 4

The Fortran wrapper:

.. literalinclude:: ../regression/reference/clibrary/wrapfclibrary.f
   :language: fortran
   :start-after: start implied_text_len
   :end-before: end implied_text_len
   :dedent: 4

Fortran usage:

.. code-block:: fortran

    character(MAXNAME) name1
    call implied_text_len(name1)
    call assert_equals("ImpliedTextLen", name1)

.. ############################################################

.. _example_acceptCharArrayIn:

acceptCharArrayIn
^^^^^^^^^^^^^^^^^

Arguments of type ``char **`` are assumed to be a list of ``NULL``
terminated strings.  In Fortran this pattern would be an array of
``CHARACTER`` where all strings are the same length.  The Fortran
variable is converted into the the C version by copying the data then
releasing it at the end of the wrapper.

:file:`pointers.yaml`:

.. code-block:: yaml

    - decl: void acceptCharArrayIn(char **names +intent(in))

This is a C file which provides the bufferify function.

:file:`wrappointers.c`:

.. literalinclude:: ../regression/reference/pointers-c/wrappointers.c
   :language: c
   :start-after: start POI_acceptCharArrayIn_bufferify
   :end-before: end POI_acceptCharArrayIn_bufferify

Most of the work is done by the helper function.
This converts the Fortran array into NULL terminated strings by
copying all of the values:

.. literalinclude:: ../regression/reference/none/helpers.c
   :language: c
   :start-after: start char_array_alloc c_source
   :end-before: end char_array_alloc c_source

Fortran calls C via the following interface:

.. literalinclude:: ../regression/reference/pointers-c/wrapfpointers.f
   :language: fortran
   :start-after: start c_accept_char_array_in
   :end-before: end c_accept_char_array_in
   :dedent: 4

The Fortran wrapper:

.. literalinclude:: ../regression/reference/pointers-c/wrapfpointers.f
   :language: fortran
   :start-after: start accept_char_array_in
   :end-before: end accept_char_array_in
   :dedent: 4

Example usage:

.. code-block:: fortran

    character(10) :: in(3) = [ &
         "dog       ", &
         "cat       ", &
         "monkey    "  &
         ]
    call accept_char_array_in(in)


std::string
-----------

.. ############################################################

.. _example_acceptStringReference:

acceptStringReference
^^^^^^^^^^^^^^^^^^^^^

C++ library function in :file:`strings.c`:

.. literalinclude:: ../regression/run/strings/strings.cpp
   :language: c
   :start-after: start acceptStringReference
   :end-before: end acceptStringReference

:file:`strings.yaml`:

.. code-block:: yaml

    - decl: void acceptStringReference(std::string & arg1)

A reference defaults to *intent(inout)* and will add both the *len*
and *len_trim* annotations.

Both generated functions will convert ``arg`` into a ``std::string``,
call the function, then copy the results back into the argument.

Which will call the C wrapper:

.. literalinclude:: ../regression/reference/strings/wrapstrings.cpp
   :language: c
   :start-after: start STR_acceptStringReference
   :end-before: end STR_acceptStringReference

The C wrapper:

.. literalinclude:: ../regression/reference/strings/wrapstrings.cpp
   :language: c
   :start-after: start STR_acceptStringReference_bufferify
   :end-before: end STR_acceptStringReference_bufferify

An interface for the native C function is also created:

.. literalinclude:: ../regression/reference/strings/wrapfstrings.f
   :language: fortran
   :start-after: start c_accept_string_reference
   :end-before: end c_accept_string_reference
   :dedent: 4

Fortran calls C via the following interface:

.. literalinclude:: ../regression/reference/strings/wrapfstrings.f
   :language: fortran
   :start-after: start c_accept_string_reference_bufferify
   :end-before: end c_accept_string_reference_bufferify
   :dedent: 4

The Fortran wrapper:

.. literalinclude:: ../regression/reference/strings/wrapfstrings.f
   :language: fortran
   :start-after: start accept_string_reference
   :end-before: end accept_string_reference
   :dedent: 4

The important thing to notice is that the pure C version could do very
bad things since it does not know how much space it has to copy into.
The bufferify version knows the allocated length of the argument.
However, since the input argument is a fixed length it may be too
short for the new string value:

Fortran usage:

.. code-block:: fortran

    character(30) str
    str = "cat"
    call accept_string_reference(str)
    call assert_true( str == "catdog")


char functions
--------------

.. ############################################################

.. _example_getCharPtr1:

getCharPtr1
^^^^^^^^^^^

.. fc_statememnt f_char_scalar_result_allocatable

Return a pointer and convert into an ``ALLOCATABLE`` ``CHARACTER``
variable.  The Fortran application is responsible to release the
memory.  However, this may be done automatically by the Fortran
runtime.

C++ library function in :file:`strings.cpp`:

.. literalinclude:: ../regression/run/strings/strings.cpp
   :language: c
   :start-after: start getCharPtr1
   :end-before: end getCharPtr1

:file:`strings.yaml`:

.. code-block:: yaml

    - decl: const char * getCharPtr1()

The C wrapper copies all of the metadata into a ``SHROUD_array``
struct which is used by the Fortran wrapper:

.. literalinclude:: ../regression/reference/strings/wrapstrings.cpp
   :language: c
   :start-after: start STR_getCharPtr1_bufferify
   :end-before: end STR_getCharPtr1_bufferify

Fortran calls C via the following interface:

.. literalinclude:: ../regression/reference/strings/wrapfstrings.f
   :language: fortran
   :start-after: start c_get_char_ptr1_bufferify
   :end-before: end c_get_char_ptr1_bufferify
   :dedent: 4

The Fortran wrapper uses the metadata in ``DSHF_rv`` to allocate
a ``CHARACTER`` variable of the correct length.
The helper function ``SHROUD_copy_string_and_free`` will copy 
the results of the C++ function into the return variable:

.. literalinclude:: ../regression/reference/strings/wrapfstrings.f
   :language: fortran
   :start-after: start get_char_ptr1
   :end-before: end get_char_ptr1
   :dedent: 4

Fortran usage:

.. code-block:: fortran

    character(len=:), allocatable :: str
    str = get_char_ptr1()

.. ############################################################

.. _example_getCharPtr2:

getCharPtr2
^^^^^^^^^^^

If you know the maximum size of string that you expect the function to
return, then the *len* attribute is used to declare the length.  The
explicit ``ALLOCATE`` is avoided but any result which is longer than
the length will be silently truncated.

C++ library function in :file:`strings.cpp`:

.. literalinclude:: ../regression/run/strings/strings.cpp
   :language: c
   :start-after: start getCharPtr2
   :end-before: end getCharPtr2

:file:`strings.yaml`:

.. code-block:: yaml

    - decl: const char * getCharPtr2() +len(30)

The C wrapper:

.. literalinclude:: ../regression/reference/strings/wrapstrings.cpp
   :language: c
   :start-after: start STR_getCharPtr2_bufferify
   :end-before: end STR_getCharPtr2_bufferify

Fortran calls C via the following interface:

.. literalinclude:: ../regression/reference/strings/wrapfstrings.f
   :language: fortran
   :start-after: start c_get_char_ptr2_bufferify
   :end-before: end c_get_char_ptr2_bufferify
   :dedent: 4

The Fortran wrapper:

.. literalinclude:: ../regression/reference/strings/wrapfstrings.f
   :language: fortran
   :start-after: start get_char_ptr2
   :end-before: end get_char_ptr2
   :dedent: 4

Fortran usage:

.. code-block:: fortran

    character(30) str
    str = get_char_ptr2()

.. ############################################################

.. _example_getCharPtr3:

getCharPtr3
^^^^^^^^^^^

Create a Fortran subroutine with an additional ``CHARACTER``
argument for the C function result. Any size character string can
be returned limited by the size of the Fortran argument.  The
argument is defined by the *F_string_result_as_arg* format string.

C++ library function in :file:`strings.cpp`:

.. literalinclude:: ../regression/run/strings/strings.cpp
   :language: c
   :start-after: start getCharPtr3
   :end-before: end getCharPtr3

:file:`strings.yaml`:

.. code-block:: yaml

    - decl: const char * getCharPtr3()
      format:
        F_string_result_as_arg: output

The C wrapper:

.. literalinclude:: ../regression/reference/strings/wrapstrings.cpp
   :language: c
   :start-after: start STR_getCharPtr3_bufferify
   :end-before: end STR_getCharPtr3_bufferify

Fortran calls C via the following interface:

.. literalinclude:: ../regression/reference/strings/wrapfstrings.f
   :language: fortran
   :start-after: start c_get_char_ptr3_bufferify
   :end-before: end c_get_char_ptr3_bufferify
   :dedent: 4

The Fortran wrapper:

.. literalinclude:: ../regression/reference/strings/wrapfstrings.f
   :language: fortran
   :start-after: start get_char_ptr3
   :end-before: end get_char_ptr3
   :dedent: 4

Fortran usage:

.. code-block:: fortran

    character(30) str
    call get_char_ptrs(str)

string functions
----------------

.. ############################################################

.. _example_getConstStringRefPure:

getConstStringRefPure
^^^^^^^^^^^^^^^^^^^^^

C++ library function in :file:`strings.cpp`:

.. literalinclude:: ../regression/run/strings/strings.cpp
   :language: c
   :start-after: start getConstStringRefPure
   :end-before: end getConstStringRefPure

:file:`strings.yaml`:

.. code-block:: yaml

    - decl: const string& getConstStringRefPure()

The C wrapper:

.. literalinclude:: ../regression/reference/strings/wrapstrings.cpp
   :language: c
   :start-after: start STR_getConstStringRefPure_bufferify
   :end-before: end STR_getConstStringRefPure_bufferify

The native C wrapper:

.. literalinclude:: ../regression/reference/strings/wrapstrings.cpp
   :language: c
   :start-after: start STR_getConstStringRefPure
   :end-before: end STR_getConstStringRefPure

Fortran calls C via the following interface:

.. literalinclude:: ../regression/reference/strings/wrapfstrings.f
   :language: fortran
   :start-after: start c_get_const_string_ref_pure_bufferify
   :end-before: end c_get_const_string_ref_pure_bufferify
   :dedent: 4

The Fortran wrapper:

.. literalinclude:: ../regression/reference/strings/wrapfstrings.f
   :language: fortran
   :start-after: start get_const_string_ref_pure
   :end-before: end get_const_string_ref_pure
   :dedent: 4

Fortran usage:

.. code-block:: fortran

    str = 'XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX'
    str = get_const_string_ref_pure()
    call assert_true( str == static_str, "getConstStringRefPure")


std::vector
-----------

.. ############################################################

.. _example_vector_sum:

vector_sum
^^^^^^^^^^

C++ library function in :file:`vectors.cpp`:

.. literalinclude:: ../regression/run/vectors/vectors.cpp
   :language: c
   :start-after: start vector_sum
   :end-before: end vector_sum

:file:`vectors.yaml`:

.. code-block:: yaml

    - decl: int vector_sum(const std::vector<int> &arg)

``intent(in)`` is implied for the *vector_sum* argument since it is
``const``.  The Fortran wrapper passes the array and the size to C.

The C wrapper:

.. literalinclude:: ../regression/reference/vectors/wrapvectors.cpp
   :language: c
   :start-after: start VEC_vector_sum
   :end-before: end VEC_vector_sum

Fortran calls C via the following interface:

.. literalinclude:: ../regression/reference/vectors/wrapfvectors.f
   :language: fortran
   :start-after: start c_vector_sum
   :end-before: end c_vector_sum
   :dedent: 4

The Fortran wrapper:

.. literalinclude:: ../regression/reference/vectors/wrapfvectors.f
   :language: fortran
   :start-after: start vector_sum
   :end-before: end vector_sum
   :dedent: 4

Fortran usage:

.. code-block:: fortran

    integer(C_INT) intv(5)
    intv = [1,2,3,4,5]
    irv = vector_sum(intv)
    call assert_true(irv .eq. 15)

.. ############################################################

.. _example_vector_iota_out:

vector_iota_out
^^^^^^^^^^^^^^^

C++ library function in :file:`vectors.cpp` accepts an empty vector
then fills in some values.
In this example, a Fortran array is passed in and will be filled.

.. literalinclude:: ../regression/run/vectors/vectors.cpp
   :language: c
   :start-after: start vector_iota_out
   :end-before: end vector_iota_out

:file:`vectors.yaml`:

.. code-block:: yaml

    - decl: void vector_iota_out(std::vector<int> &arg+intent(out))

The C wrapper allocates a new ``std::vector`` instance which will be
returned to the Fortran wrapper.
Variable ``Darg`` will be filled with the meta data for the ``std::vector``
in a form that allows Fortran to access it.
The value of ``Darg->cxx.idtor`` is computed by Shroud and used
to release the memory (index of destructor).

.. literalinclude:: ../regression/reference/vectors/wrapvectors.cpp
   :language: c
   :start-after: start VEC_vector_iota_out_bufferify
   :end-before: end VEC_vector_iota_out_bufferify

Fortran calls C via the following interface:

.. literalinclude:: ../regression/reference/vectors/wrapfvectors.f
   :language: fortran
   :start-after: start c_vector_iota_out_bufferify
   :end-before: end c_vector_iota_out_bufferify
   :dedent: 4

The Fortran wrapper passes a ``SHROUD_array`` instance which will be
filled by the C wrapper.

.. literalinclude:: ../regression/reference/vectors/wrapfvectors.f
   :language: fortran
   :start-after: start vector_iota_out
   :end-before: end vector_iota_out
   :dedent: 4

Function ``SHROUD_copy_array_int`` copies the values
into the user's argument.
If the argument is too short, not all values returned by
the library function will be copied.

.. literalinclude:: ../regression/reference/vectors/utilvectors.cpp
   :language: c
   :start-after: start helper copy_array
   :end-before: end helper copy_array

Finally, the ``std::vector`` is released based on the value of ``idtor``:

.. literalinclude:: ../regression/reference/vectors/utilvectors.cpp
   :language: c
   :start-after: start release allocated memory
   :end-before: end release allocated memory

Fortran usage:

.. code-block:: fortran

    integer(C_INT) intv(5)
    intv(:) = 0
    call vector_iota_out(intv)
    call assert_true(all(intv(:) .eq. [1,2,3,4,5]))

.. ############################################################

.. _example_vector_iota_out_alloc:

vector_iota_out_alloc
^^^^^^^^^^^^^^^^^^^^^

C++ library function in :file:`vectors.cpp` accepts an empty vector
then fills in some values.
In this example, the Fortran argument is ``ALLOCATABLE`` and will
be sized based on the output of the library function.

.. literalinclude:: ../regression/run/vectors/vectors.cpp
   :language: c
   :start-after: start vector_iota_out_alloc
   :end-before: end vector_iota_out_alloc

The attribute *+deref(allocatable)* will cause the argument to be an
``ALLOCATABLE`` array.

:file:`vectors.yaml`:

.. code-block:: yaml

    - decl: void vector_iota_out_alloc(std::vector<int> &arg+intent(out)+deref(allocatable))

The C wrapper:

.. literalinclude:: ../regression/reference/vectors/wrapvectors.cpp
   :language: c
   :start-after: start VEC_vector_iota_out_alloc_bufferify
   :end-before: end VEC_vector_iota_out_alloc_bufferify

Fortran calls C via the following interface:

.. literalinclude:: ../regression/reference/vectors/wrapfvectors.f
   :language: fortran
   :start-after: start c_vector_iota_out_alloc_bufferify
   :end-before: end c_vector_iota_out_alloc_bufferify
   :dedent: 4

The Fortran wrapper passes a ``SHROUD_array`` instance which will be
filled by the C wrapper.
After the function returns, the ``allocate`` statement allocates an 
array of the proper length.

.. literalinclude:: ../regression/reference/vectors/wrapfvectors.f
   :language: fortran
   :start-after: start vector_iota_out_alloc
   :end-before: end vector_iota_out_alloc
   :dedent: 4

``inta`` is ``intent(out)``, so it will be deallocated upon entry to ``vector_iota_out_alloc``.

Fortran usage:

.. code-block:: fortran

    integer(C_INT), allocatable :: inta(:)
    call vector_iota_out_alloc(inta)
    call assert_true(allocated(inta))
    call assert_equals(5 , size(inta))
    call assert_true( all(inta == [1,2,3,4,5]), &
         "vector_iota_out_alloc value")


.. ############################################################

.. _example_vector_iota_inout_alloc:

vector_iota_inout_alloc
^^^^^^^^^^^^^^^^^^^^^^^

C++ library function in :file:`vectors.cpp`:

.. literalinclude:: ../regression/run/vectors/vectors.cpp
   :language: c
   :start-after: start vector_iota_inout_alloc
   :end-before: end vector_iota_inout_alloc

:file:`vectors.yaml`:

.. code-block:: yaml

    - decl: void vector_iota_out_alloc(std::vector<int> &arg+intent(inout)+deref(allocatable))

The C wrapper creates a new ``std::vector`` and initializes it to the
Fortran argument.

.. literalinclude:: ../regression/reference/vectors/wrapvectors.cpp
   :language: c
   :start-after: start VEC_vector_iota_inout_alloc_bufferify
   :end-before: end VEC_vector_iota_inout_alloc_bufferify

Fortran calls C via the following interface:

.. literalinclude:: ../regression/reference/vectors/wrapfvectors.f
   :language: fortran
   :start-after: start c_vector_iota_inout_alloc_bufferify
   :end-before: end c_vector_iota_inout_alloc_bufferify
   :dedent: 4

The Fortran wrapper will deallocate the argument after returning
since it is *intent(inout)*.  The *in* values are now stored in
the ``std::vector``.  A new array is allocated to the current size
of the ``std::vector``.  Fortran has no reallocate statement.
Finally, the new values are copied into the Fortran array and
the ``std::vector`` is released.

.. literalinclude:: ../regression/reference/vectors/wrapfvectors.f
   :language: fortran
   :start-after: start vector_iota_inout_alloc
   :end-before: end vector_iota_inout_alloc
   :dedent: 4


``inta`` is ``intent(inout)``, so it will NOT be deallocated upon
entry to ``vector_iota_inout_alloc``.
Fortran usage:

.. code-block:: fortran

    call vector_iota_inout_alloc(inta)
    call assert_true(allocated(inta))
    call assert_equals(10 , size(inta))
    call assert_true( all(inta == [1,2,3,4,5,11,12,13,14,15]), &
         "vector_iota_inout_alloc value")
    deallocate(inta)

Void Pointers
-------------

.. ############################################################

.. _example_passAssumedType:

passAssumedType
^^^^^^^^^^^^^^^

C library function in :file:`clibrary.c`:

.. literalinclude:: ../regression/run/clibrary/clibrary.c
   :language: c
   :start-after: start passAssumedType
   :end-before: end passAssumedType

:file:`clibrary.yaml`:

.. code-block:: yaml

    - decl: int passAssumedType(void *arg+assumedtype)

Fortran calls C via the following interface:

.. literalinclude:: ../regression/reference/clibrary/wrapfclibrary.f
   :language: fortran
   :start-after: start pass_assumed_type
   :end-before: end pass_assumed_type
   :dedent: 4

Fortran usage:

.. code-block:: fortran

    use iso_c_binding, only : C_INT
    integer(C_INT) rv_int
    rv_int = pass_assumed_type(23_C_INT)

As a reminder, ``23_C_INT`` creates an ``integer(C_INT)`` constant.

.. note:: Assumed-type was introduced in Fortran 2018.

.. ############################################################

.. _example_passAssumedTypeDim:

passAssumedTypeRank
^^^^^^^^^^^^^^^^^^^

C library function in :file:`clibrary.c`:

.. literalinclude:: ../regression/run/clibrary/clibrary.c
   :language: c
   :start-after: start passAssumedTypeRank
   :end-before: end passAssumedTypeRank

:file:`clibrary.yaml`:

.. code-block:: yaml

    - decl: int passAssumedTypeRank(void *arg+assumedtype+rank(1))

Fortran calls C via the following interface:

.. literalinclude:: ../regression/reference/clibrary/wrapfclibrary.f
   :language: fortran
   :start-after: start pass_assumed_type_rank
   :end-before: end pass_assumed_type_rank
   :dedent: 4

Example usage:

.. code-block:: fortran

    use iso_c_binding, only : C_INT, C_DOUBLE
    integer(C_INT) int_array(10)
    real(C_DOUBLE) double_array(2,5)
    call pass_assumed_type_rank(int_array)
    call pass_assumed_type_rank(double_array)

.. note:: Assumed-type was introduced in Fortran 2018.

.. ############################################################

.. _example_passVoidStarStar:

passVoidStarStar
^^^^^^^^^^^^^^^^

C library function in :file:`clibrary.c`:

.. literalinclude:: ../regression/run/clibrary/clibrary.c
   :language: c
   :start-after: start passVoidStarStar
   :end-before: end passVoidStarStar

:file:`clibrary.yaml`:

.. code-block:: yaml

    - decl: void passVoidStarStar(void *in+intent(in), void **out+intent(out))

Fortran calls C via the following interface:

.. literalinclude:: ../regression/reference/clibrary/wrapfclibrary.f
   :language: fortran
   :start-after: start pass_void_star_star
   :end-before: end pass_void_star_star
   :dedent: 4

Example usage:

.. code-block:: fortran

    use iso_c_binding, only : C_INT, C_NULL_PTR, c_associated
    integer(C_INT) int_var
    cptr1 = c_loc(int_var)
    cptr2 = C_NULL_PTR
    call pass_void_star_star(cptr1, cptr2)
    call assert_true(c_associated(cptr1, cptr2))


Function Pointers
-----------------

.. ############################################################

.. _example_callback1:

callback1
^^^^^^^^^

C++ library function in :file:`tutorial.cpp`:

.. literalinclude:: ../regression/run/tutorial/tutorial.cpp
   :language: c
   :start-after: start callback1
   :end-before: end callback1

:file:`tutorial.yaml`:

.. code-block:: yaml

    - decl: int callback1(int in, int (*incr)(int));

The C wrapper:

.. literalinclude:: ../regression/reference/tutorial/wrapTutorial.cpp
   :language: c
   :start-after: start TUT_callback1
   :end-before: end TUT_callback1

Creates the abstract interface:

.. literalinclude:: ../regression/reference/tutorial/wrapftutorial.f
   :language: fortran
   :start-after: start abstract callback1_incr
   :end-before: end abstract callback1_incr
   :dedent: 4

Fortran calls C via the following interface:

.. literalinclude:: ../regression/reference/tutorial/wrapftutorial.f
   :language: fortran
   :start-after: start callback1
   :end-before: end callback1
   :dedent: 4

Fortran usage:

.. code-block:: fortran

    module worker
      use iso_c_binding
    contains
      subroutine userincr(i) bind(C)
        integer(C_INT), value :: i
        ! do work of callback
      end subroutine user

      subroutine work
        call callback1(1, userincr)
      end subroutine work
    end module worker


.. ############################################################

.. _example_callback1_funptr:

callback1_funptr
^^^^^^^^^^^^^^^^

C library function in :file:`funptr.c`. The actual function would need
some way to know the interface/prototype of the function that was
passed in. Perhaps by another argument or some other state:

.. literalinclude:: ../regression/run/funptr/funptr.c
   :language: c
   :start-after: start callback1_funptr
   :end-before: end callback1_funptr

:file:`funptr.yaml`:

.. code-block:: yaml

    - decl: void callback1_funptr(void (*incr)(void)+funptr)

The Fortran wrapper.
By using ``funptr`` no abstract interface is used:

.. literalinclude:: ../regression/reference/funptr-c/wrapffunptr.f
   :language: fortran
   :start-after: start callback1_funptr
   :end-before: end callback1_funptr
   :dedent: 4

Fortran usage:

.. code-block:: fortran

    module worker
      use iso_c_binding
    contains
      subroutine userincr_int() bind(C)
        ! do work of callback
      end subroutine user_int

      subroutine userincr_double() bind(C)
        ! do work of callback
      end subroutine user_int

      subroutine work
        call callback1_funptr(c_funloc(userincr_int))
        call callback1_funptr(c_funloc(userincr_double))
      end subroutine work
    end module worker

Struct
------

Struct creating is described in :ref:`Fortran Structs <struct_fortran>`.


.. ############################################################

.. _example_passStruct1:

passStruct1
^^^^^^^^^^^

C library function in :file:`struct.c`:

.. literalinclude:: ../regression/run/struct/struct.c
   :language: c
   :start-after: start passStruct1
   :end-before: end passStruct1

:file:`struct.yaml`:

.. code-block:: yaml

    - decl: int passStruct1(Cstruct1 *s1)

Fortran calls C via the following interface:

.. literalinclude:: ../regression/reference/struct-c/wrapfstruct.f
   :language: fortran
   :start-after: start pass_struct1
   :end-before: end pass_struct1
   :dedent: 4

Fortran usage:

.. code-block:: fortran

    type(cstruct1) str1
    str1%ifield = 12
    str1%dfield = 12.6
    call assert_equals(12, pass_struct1(str1), "passStruct1")


.. ############################################################

.. _example_passStructByValue:

passStructByValue
^^^^^^^^^^^^^^^^^

C library function in :file:`struct.c`:

.. literalinclude:: ../regression/run/struct/struct.c
   :language: c
   :start-after: start passStructByValue
   :end-before: end passStructByValue

:file:`struct.yaml`:

.. code-block:: yaml

    - decl: double passStructByValue(struct1 arg)

Fortran calls C via the following interface:

.. literalinclude:: ../regression/reference/struct-c/wrapfstruct.f
   :language: fortran
   :start-after: start pass_struct_by_value
   :end-before: end pass_struct_by_value
   :dedent: 4

Fortran usage:

.. code-block:: fortran

    type(cstruct1) str1
    str1%ifield = 2_C_INT
    str1%dfield = 2.0_C_DOUBLE
    rvi = pass_struct_by_value(str1)
    call assert_equals(4, rvi, "pass_struct_by_value")
    ! Make sure str1 was passed by value.
    call assert_equals(2_C_INT, str1%ifield, "pass_struct_by_value ifield")
    call assert_equals(2.0_C_DOUBLE, str1%dfield, "pass_struct_by_value dfield")


Class Type
----------

.. ############################################################

.. _example_constructor_and_destructor:

constructor and destructor
^^^^^^^^^^^^^^^^^^^^^^^^^^

The C++ header file from :file:`classes.hpp`.

.. code-block:: c++

    class Class1
    {
    public:
        int m_flag;
        int m_test;
        Class1()         : m_flag(0), m_test(0)    {};
        Class1(int flag) : m_flag(flag), m_test(0) {};
    };

:file:`classes.yaml`:

.. code-block:: yaml

    declarations:
    - decl: class Class1
      declarations:
      - decl: Class1()
        format:
          function_suffix: _default
      - decl: Class1(int flag)
        format:
        function_suffix: _flag
      - decl: ~Class1() +name(delete)

A C wrapper function is created for each constructor and the destructor.

The C wrappers:

.. literalinclude:: ../regression/reference/classes/wrapClass1.cpp
   :language: c
   :start-after: start CLA_Class1_ctor_default
   :end-before: end CLA_Class1_ctor_default

.. literalinclude:: ../regression/reference/classes/wrapClass1.cpp
   :language: c
   :start-after: start CLA_Class1_ctor_flag
   :end-before: end CLA_Class1_ctor_flag

.. literalinclude:: ../regression/reference/classes/wrapClass1.cpp
   :language: c
   :start-after: start CLA_Class1_delete
   :end-before: end CLA_Class1_delete

The corresponding Fortran interfaces:

.. literalinclude:: ../regression/reference/classes/wrapfclasses.f
   :language: fortran
   :start-after: start c_class1_ctor_default
   :end-before: end c_class1_ctor_default
   :dedent: 4

.. literalinclude:: ../regression/reference/classes/wrapfclasses.f
   :language: fortran
   :start-after: start c_class1_ctor_flag
   :end-before: end c_class1_ctor_flag
   :dedent: 4

.. literalinclude:: ../regression/reference/classes/wrapfclasses.f
   :language: fortran
   :start-after: start c_class1_delete
   :end-before: end c_class1_delete
   :dedent: 4

And the Fortran wrappers:

.. literalinclude:: ../regression/reference/classes/wrapfclasses.f
   :language: fortran
   :start-after: start class1_ctor_default
   :end-before: end class1_ctor_default
   :dedent: 4

.. literalinclude:: ../regression/reference/classes/wrapfclasses.f
   :language: fortran
   :start-after: start class1_ctor_flag
   :end-before: end class1_ctor_flag
   :dedent: 4

.. literalinclude:: ../regression/reference/classes/wrapfclasses.f
   :language: fortran
   :start-after: start class1_delete
   :end-before: end class1_delete
   :dedent: 4

The Fortran shadow class adds the type-bound method for the destructor:

.. code-block:: fortran

    type, bind(C) :: SHROUD_class1_capsule
        type(C_PTR) :: addr = C_NULL_PTR  ! address of C++ memory
        integer(C_INT) :: idtor = 0       ! index of destructor
    end type SHROUD_class1_capsule

    type class1
        type(SHROUD_class1_capsule) :: cxxmem
    contains
        procedure :: delete => class1_delete
    end type class1

The constructors are not type-bound procedures. But they
are combined into a generic interface.

.. literalinclude:: ../regression/reference/classes/wrapfclasses.f
   :language: fortran
   :start-after: ! start generic interface class1
   :end-before: ! end generic interface class1
   :dedent: 4

A class instance is created and destroy from Fortran as:

.. code-block:: fortran

    use classes_mod
    type(class1) obj

    obj = class1()
    call obj%delete

Corresponding C++ code:

.. code-block:: c++

    include <classes.hpp>

    classes::Class1 * obj = new classes::Class1;

    delete obj;

.. ############################################################

.. _example_getter_and_setter:

Getter and Setter
^^^^^^^^^^^^^^^^^

The C++ header file from :file:`classes.hpp`.

.. code-block:: c++

    class Class1
    {
    public:
        int m_flag;
        int m_test;
    };

:file:`classes.yaml`:

.. code-block:: yaml

    declarations:
    - decl: class Class1
      declarations:
      - decl: int m_flag +readonly;
      - decl: int m_test +name(test);

A C wrapper function is created for each getter and setter.
If the *readonly* attribute is added, then only a getter is created.
In this case ``m_`` is a convention used to designate member variables.
The Fortran attribute is renamed as **test** to avoid cluttering
the Fortran API with this convention.

The C wrappers:

.. literalinclude:: ../regression/reference/classes/wrapClass1.cpp
   :language: c
   :start-after: start CLA_Class1_get_m_flag
   :end-before: end CLA_Class1_get_m_flag

.. literalinclude:: ../regression/reference/classes/wrapClass1.cpp
   :language: c
   :start-after: start CLA_Class1_get_test
   :end-before: end CLA_Class1_get_test

.. literalinclude:: ../regression/reference/classes/wrapClass1.cpp
   :language: c
   :start-after: start CLA_Class1_set_test
   :end-before: end CLA_Class1_set_test

The corresponding Fortran interfaces:

.. literalinclude:: ../regression/reference/classes/wrapfclasses.f
   :language: fortran
   :start-after: start c_class1_get_m_flag
   :end-before: end c_class1_get_m_flag
   :dedent: 4

.. literalinclude:: ../regression/reference/classes/wrapfclasses.f
   :language: fortran
   :start-after: start c_class1_get_test
   :end-before: end c_class1_get_test
   :dedent: 4

.. literalinclude:: ../regression/reference/classes/wrapfclasses.f
   :language: fortran
   :start-after: start c_class1_set_test
   :end-before: end c_class1_set_test
   :dedent: 4

And the Fortran wrappers:

.. literalinclude:: ../regression/reference/classes/wrapfclasses.f
   :language: fortran
   :start-after: start class1_get_m_flag
   :end-before: end class1_get_m_flag
   :dedent: 4

.. literalinclude:: ../regression/reference/classes/wrapfclasses.f
   :language: fortran
   :start-after: start class1_get_test
   :end-before: end class1_get_test
   :dedent: 4

.. literalinclude:: ../regression/reference/classes/wrapfclasses.f
   :language: fortran
   :start-after: start class1_set_test
   :end-before: end class1_set_test
   :dedent: 4

The Fortran shadow class adds the type-bound methods:

.. code-block:: fortran

    type class1
        type(SHROUD_class1_capsule) :: cxxmem
    contains
        procedure :: get_m_flag => class1_get_m_flag
        procedure :: get_test => class1_get_test
        procedure :: set_test => class1_set_test
    end type class1

The class variables can be used as:

.. code-block:: fortran

    use classes_mod
    type(class1) obj
    integer iflag

    obj = class1()
    call obj%set_test(4)
    iflag = obj%get_test()

Corresponding C++ code:

.. code-block:: c++

    include <classes.hpp>
    classes::Class1 obj = new * classes::Class1;
    obj->m_test = 4;
    int iflag = obj->m_test;

.. ############################################################

.. _example_struct_as_class:

Struct as a Class
^^^^^^^^^^^^^^^^^

While C does not support object-oriented programming directly, it can be
emulated by using structs.  The 'base class' struct is ``Cstruct_as_clss``.
It is subclassed by ``Cstruct_as_subclass`` which explicitly duplicates
the members of ``C_struct_as_class``.
The C header file from :file:`struct.h`.

.. literalinclude:: ../regression/run/struct/struct.h
   :language: c
   :start-after: start struct Cstruct_as_class
   :end-before: end struct Cstruct_as_class

The C 'constructor' returns a pointer to an instance of the object.

.. literalinclude:: ../regression/run/struct/struct.h
   :language: c
   :start-after: start Cstruct_as_class ctor
   :end-before: end Cstruct_as_class ctor

The 'methods' pass an instance of the class as an explicit *this* object.
          
.. literalinclude:: ../regression/run/struct/struct.h
   :language: c
   :start-after: start Cstruct_as_class Cstruct_as_class_sum
   :end-before: end Cstruct_as_class Cstruct_as_class_sum

The methods are wrapped in :file:`classes.yaml`:

.. code-block:: yaml

    declarations:
    - decl: struct Cstruct_as_class {
              int x1;
              int y1;
            };
      options:
        wrap_struct_as: class
    
    - decl: Cstruct_as_class *Create_Cstruct_as_class(void)
      options:
        class_ctor: Cstruct_as_class
    - decl: Cstruct_as_class *Create_Cstruct_as_class_args(int x, int y)
      options:
        class_ctor: Cstruct_as_class
    
    - decl: int Cstruct_as_class_sum(const Cstruct_as_class *point +pass)
      options:
        class_method: Cstruct_as_class
      format:
        F_name_function: sum

    - decl: struct Cstruct_as_subclass {
              int x1;
              int y1;
              int z1;
            };
      options:
        wrap_struct_as: class
        class_baseclass: Cstruct_as_class
    - decl: Cstruct_as_subclass *Create_Cstruct_as_subclass_args(int x, int y, int z)
      options:
        wrap_python: False
        class_ctor: Cstruct_as_subclass

This uses several options to creates the class features for the struct:
*wrap_struct_as*, *class_ctor*, *class_method*.

.. literalinclude:: ../regression/reference/struct-c/wrapfstruct.f
   :language: c
   :start-after: start derived-type cstruct_as_class
   :end-before: end derived-type cstruct_as_class

The subclass is created using the Fortran ``EXTENDS`` keyword.  No
additional members are added. The ``cxxmem`` field from
``cstruct_as_class`` will now point to an instance of the C struct
``Cstruct_as_subclass``.

.. literalinclude:: ../regression/reference/struct-c/wrapfstruct.f
   :language: c
   :start-after: start derived-type cstruct_as_subclass
   :end-before: end derived-type cstruct_as_subclass

The C wrapper to construct the struct-as-class.  It calls the C function
and fills in the fields for the shadow struct.

.. literalinclude:: ../regression/reference/struct-c/wrapstruct.c
   :language: c
   :start-after: start STR_Create_Cstruct_as_class
   :end-before: end STR_Create_Cstruct_as_class

A Fortran generic interface is created for the class:

.. literalinclude:: ../regression/reference/struct-c/wrapfstruct.f
   :language: fortran
   :start-after: start generic interface cstruct_as_class
   :end-before: end generic interface cstruct_as_class
   :dedent: 4

And the Fortran constructor call the C wrapper function.

.. literalinclude:: ../regression/reference/struct-c/wrapfstruct.f
   :language: fortran
   :start-after: start create_cstruct_as_class
   :end-before: end create_cstruct_as_class
   :dedent: 4

The class can be used as:


.. literalinclude:: ../regression/run/struct/main.f
   :language: fortran
   :start-after: start main.f test_struct_class
   :end-before: end main.f test_struct_class
   :dedent: 4

.. ############################################################

.. _example_UseDefaultArguments:

Default Value Arguments
-----------------------

The default values are provided in the function declaration.

C++ library function in :file:`tutorial.cpp`:

.. literalinclude:: ../regression/run/tutorial/tutorial.hpp
   :language: c++
   :start-after: start UseDefaultArguments
   :end-before: end UseDefaultArguments

:file:`tutorial.yaml`:

.. code-block:: yaml

  - decl: double UseDefaultArguments(double arg1 = 3.1415, bool arg2 = true)
    default_arg_suffix:
    -  
    -  _arg1
    -  _arg1_arg2

A C++ wrapper is created which calls the C++ function with no arguments 
with default values and then adds a wrapper with an explicit argument
for each default value argument. In this case, three wrappers are created.
Since the C++ compiler provides the default value, it is necessary to 
create each wrapper.

:file:`wrapTutorial.cpp`:

.. literalinclude:: ../regression/reference/tutorial/wrapTutorial.cpp
   :language: c++
   :start-after: start TUT_UseDefaultArguments
   :end-before: end TUT_UseDefaultArguments

.. literalinclude:: ../regression/reference/tutorial/wrapTutorial.cpp
   :language: c++
   :start-after: start TUT_UseDefaultArguments_arg1
   :end-before: end TUT_UseDefaultArguments_arg1

.. literalinclude:: ../regression/reference/tutorial/wrapTutorial.cpp
   :language: c++
   :start-after: start TUT_UseDefaultArguments_arg1_arg2
   :end-before: end TUT_UseDefaultArguments_arg1_arg2

This creates three corresponding Fortran interfaces:

.. literalinclude:: ../regression/reference/tutorial/wrapftutorial.f
   :language: fortran
   :start-after: start c_use_default_arguments
   :end-before: end c_use_default_arguments
   :dedent: 4

.. literalinclude:: ../regression/reference/tutorial/wrapftutorial.f
   :language: fortran
   :start-after: start c_use_default_arguments_arg1
   :end-before: end c_use_default_arguments_arg1
   :dedent: 4

.. literalinclude:: ../regression/reference/tutorial/wrapftutorial.f
   :language: fortran
   :start-after: start c_use_default_arguments_arg1_arg2
   :end-before: end c_use_default_arguments_arg1_arg2
   :dedent: 4

In many case the interfaces would be enough to call the routines.
However, in order to have a generic interface, there must be
explicit Fortran wrappers:

.. literalinclude:: ../regression/reference/tutorial/wrapftutorial.f
   :language: fortran
   :start-after: start use_default_arguments
   :end-before: end use_default_arguments
   :dedent: 4

.. literalinclude:: ../regression/reference/tutorial/wrapftutorial.f
   :language: fortran
   :start-after: start use_default_arguments_arg1
   :end-before: end use_default_arguments_arg1
   :dedent: 4

.. literalinclude:: ../regression/reference/tutorial/wrapftutorial.f
   :language: fortran
   :start-after: start use_default_arguments_arg1_arg2
   :end-before: end use_default_arguments_arg1_arg2
   :dedent: 4

The Fortran generic interface adds the ability to call any of the
functions by the C++ function name:

.. code-block:: fortran

    interface use_default_arguments
        module procedure use_default_arguments
        module procedure use_default_arguments_arg1
        module procedure use_default_arguments_arg1_arg2
    end interface use_default_arguments

Usage:

.. code-block:: fortran

    real(C_DOUBLE) rv
    rv = use_default_arguments()
    rv = use_default_arguments(1.d0)
    rv = use_default_arguments(1.d0, .false.)

.. ############################################################

.. _example_GenericReal:

Generic Real
------------

C library function in :file:`clibrary.c`:

.. literalinclude:: ../regression/run/generic/generic.c
   :language: c
   :start-after: start GenericReal
   :end-before: end GenericReal

:file:`generic.yaml`:

.. code-block:: yaml

    - decl: void GenericReal(double arg)
      fortran_generic:
      - decl: (float arg)
        function_suffix: float
      - decl: (double arg)
        function_suffix: double

Fortran calls C via the following interface:

.. literalinclude:: ../regression/reference/generic/wrapfgeneric.f
   :language: fortran
   :start-after: start c_generic_real
   :end-before: end c_generic_real
   :dedent: 4

There is a single interface since there is a single C function.
A generic interface is created for each declaration in the *fortran_generic* block.

.. literalinclude:: ../regression/reference/generic/wrapfgeneric.f
   :language: fortran
   :start-after: ! start generic interface generic_real
   :end-before: ! end generic interface generic_real
   :dedent: 4

A Fortran wrapper is created for each declaration in the *fortran_generic* block.
The argument is explicitly converted to a ``C_DOUBLE`` before calling the C function
in ``generic_real_float``.  There is no conversion necessary in ``generic_real_double``.

.. literalinclude:: ../regression/reference/generic/wrapfgeneric.f
   :language: fortran
   :start-after: start generic_real_float
   :end-before: end generic_real_float
   :dedent: 4

.. literalinclude:: ../regression/reference/generic/wrapfgeneric.f
   :language: fortran
   :start-after: start generic_real_double
   :end-before: end generic_real_double
   :dedent: 4

The function can be called via the generic interface with either type.
If the specific function is called, the correct type must be passed.

.. code-block:: fortran

    call generic_real(0.0)
    call generic_real(0.0d0)

    call generic_real_float(0.0)
    call generic_real_double(0.0d0)

In C, the compiler will promote the argument.

.. code-block:: c

    GenericReal(0.0f);
    GenericReal(0.0);
