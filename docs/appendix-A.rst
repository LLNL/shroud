.. Copyright (c) 2017-2019, Lawrence Livermore National Security, LLC and
   other Shroud Project Developers.
   See the top-level COPYRIGHT file for details.

   SPDX-License-Identifier: (BSD-3-Clause)

.. All of the examples are ordered as
   C or C++ code from users library
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
   :start-after: start TUT_no_return_no_arguments
   :end-before: end TUT_no_return_no_arguments

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

C library function:

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

C library function:

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

.. literalinclude:: ../regression/run/pointers/pointers.cpp
   :language: c
   :start-after: start Sum
   :end-before: end Sum

:file:`pointers.yaml`:

.. code-block:: yaml

   - decl: void Sum(int len +implied(size(values)),
                    int *values +dimension(:)+intent(in),
                    int *result +intent(out))

The ``POI`` prefix to the function names is derived from 
the format field *C_prefix* which defaults to the first three letters
of the *library* field, in this case *pointers*.
This is a C++ file which provides a C API via ``extern "C"``.
:file:`wrappointers.cpp`:

.. literalinclude:: ../regression/reference/pointers/wrappointers.cpp
   :language: c
   :start-after: start POI_sum
   :end-before: end POI_sum

Fortran calls C via the following interface:

.. literalinclude:: ../regression/reference/pointers/wrapfpointers.f
   :language: fortran
   :start-after: start c_sum
   :end-before: end c_sum
   :dedent: 4

The Fortran wrapper:

.. literalinclude:: ../regression/reference/pointers/wrapfpointers.f
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

.. _example_getMinMax:

getMinMax
^^^^^^^^^

No Fortran function is created.  Only an interface to a C wrapper
which dereference the pointers so they can be treated as references.

C++ library function:

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
   :start-after: start TUT_get_min_max
   :end-before: end TUT_get_min_max

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

Bool
----

.. ############################################################

.. _example_checkBool:

checkBool
^^^^^^^^^

Assignments are done in the Fortran wrapper to convert between
``logical`` and ``logical(C_BOOL)``.

C function:

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
the library variable ``MAXNAME``. From :file:`clibrary.c`:

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
   :start-after: start CLI_return_one_name_bufferify
   :end-before: end CLI_return_one_name_bufferify

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

C++ library function:

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
   :start-after: start STR_pass_char_ptr
   :end-before: end STR_pass_char_ptr

The C wrapper:

.. literalinclude:: ../regression/reference/strings/wrapstrings.cpp
   :language: c
   :start-after: start STR_pass_char_ptr_bufferify
   :end-before: end STR_pass_char_ptr_bufferify

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
   :start-after: start CLI_implied_text_len_bufferify
   :end-before: end CLI_implied_text_len_bufferify

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


std::string
-----------

.. ############################################################

.. _example_acceptStringReference:

acceptStringReference
^^^^^^^^^^^^^^^^^^^^^

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
   :start-after: start STR_accept_string_reference
   :end-before: end STR_accept_string_reference

The C wrapper:

.. literalinclude:: ../regression/reference/strings/wrapstrings.cpp
   :language: c
   :start-after: start STR_accept_string_reference_bufferify
   :end-before: end STR_accept_string_reference_bufferify

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

Return a pointer and convert into an ``ALLOCATABLE`` ``CHARACTER``
variable.  The Fortran application is responsible to release the
memory.  However, this may be done automatically by the Fortran
runtime.

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
   :start-after: start STR_get_char_ptr1_bufferify
   :end-before: end STR_get_char_ptr1_bufferify

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
   :start-after: start STR_get_char_ptr2_bufferify
   :end-before: end STR_get_char_ptr2_bufferify

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
Works with Fortran 90 and later.

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
   :start-after: start STR_get_char_ptr3_bufferify
   :end-before: end STR_get_char_ptr3_bufferify

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
   :start-after: start STR_get_const_string_ref_pure_bufferify
   :end-before: end STR_get_const_string_ref_pure_bufferify

The native C wrapper:

.. literalinclude:: ../regression/reference/strings/wrapstrings.cpp
   :language: c
   :start-after: start STR_get_const_string_ref_pure
   :end-before: end STR_get_const_string_ref_pure

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
   :start-after: start VEC_vector_sum_bufferify
   :end-before: end VEC_vector_sum_bufferify

Fortran calls C via the following interface:

.. literalinclude:: ../regression/reference/vectors/wrapfvectors.f
   :language: fortran
   :start-after: start c_vector_sum_bufferify
   :end-before: end c_vector_sum_bufferify
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

.. literalinclude:: ../regression/run/vectors/vectors.cpp
   :language: c
   :start-after: start vector_iota_out
   :end-before: end vector_iota_out

:file:`vectors.yaml`:

.. code-block:: yaml

    - decl: void vector_iota_out(std::vector<int> &arg+intent(out))

The C wrapper:

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

The Fortran wrapper:

.. literalinclude:: ../regression/reference/vectors/wrapfvectors.f
   :language: fortran
   :start-after: start vector_iota_out
   :end-before: end vector_iota_out
   :dedent: 4

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

.. literalinclude:: ../regression/run/vectors/vectors.cpp
   :language: c
   :start-after: start vector_iota_out_alloc
   :end-before: end vector_iota_out_alloc

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

The Fortran wrapper:

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

.. literalinclude:: ../regression/run/vectors/vectors.cpp
   :language: c
   :start-after: start vector_iota_inout_alloc
   :end-before: end vector_iota_inout_alloc

:file:`vectors.yaml`:

.. code-block:: yaml

    - decl: void vector_iota_out_alloc(std::vector<int> &arg+intent(inout)+deref(allocatable))

The C wrapper:

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

The Fortran wrapper:

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

C library function:

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

.. ############################################################

.. _example_passAssumedTypeDim:

passAssumedTypeDim
^^^^^^^^^^^^^^^^^^

C library function:

.. literalinclude:: ../regression/run/clibrary/clibrary.c
   :language: c
   :start-after: start passAssumedTypeDim
   :end-before: end passAssumedTypeDim

:file:`clibrary.yaml`:

.. code-block:: yaml

    - decl: int passAssumedTypeDim(void *arg+assumedtype+dimension)

Fortran calls C via the following interface:

.. literalinclude:: ../regression/reference/clibrary/wrapfclibrary.f
   :language: fortran
   :start-after: start pass_assumed_type_dim
   :end-before: end pass_assumed_type_dim
   :dedent: 4

Example usage:

.. code-block:: fortran

    use iso_c_binding, only : C_INT, C_DOUBLE
    integer(C_INT) int_array(10)
    real(C_DOUBLE) double_array(2,5)
    call pass_assumed_type_dim(int_array)
    call pass_assumed_type_dim(double_array)

.. ############################################################

.. _example_passVoidStarStar:

passVoidStarStar
^^^^^^^^^^^^^^^^

C library function:

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

C++ library function:

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

.. _example_callback1c:

callback1c
^^^^^^^^^^

C library function:

.. literalinclude:: ../regression/run/clibrary/clibrary.c
   :language: c
   :start-after: start callback1
   :end-before: end callback1

:file:`clibrary.yaml`:

.. code-block:: yaml

    - decl: int callback1(int type, void (*incr)()+external)

Creates the abstract interface:

.. literalinclude:: ../regression/reference/clibrary/wrapfclibrary.f
   :language: fortran
   :start-after: start abstract callback1_incr
   :end-before: end abstract callback1_incr
   :dedent: 4

Fortran calls C via the following interface:

.. literalinclude:: ../regression/reference/clibrary/wrapfclibrary.f
   :language: fortran
   :start-after: start c_callback1
   :end-before: end c_callback1
   :dedent: 4

.. XXX why is C_PTR used here ^

The Fortran wrapper.
By using ``external`` no abstract interface is used:

.. literalinclude:: ../regression/reference/clibrary/wrapfclibrary.f
   :language: fortran
   :start-after: start callback1
   :end-before: end callback1
   :dedent: 4

Fortran usage:

.. code-block:: fortran

    module worker
      use iso_c_binding
    contains
      subroutine userincr_int(i) bind(C)
        integer(C_INT), value :: i
        ! do work of callback
      end subroutine user_int

      subroutine userincr_double(i) bind(C)
        real(C_DOUBLE), value :: i
        ! do work of callback
      end subroutine user_int

      subroutine work
        call callback1c(1, userincr_int)
        call callback1c(1, userincr_double)
      end subrouine work
    end module worker

Struct
------

.. ############################################################

.. _example_passStruct1:

passStruct1
^^^^^^^^^^^

C library function:

.. literalinclude:: ../regression/run/struct/struct.c
   :language: c
   :start-after: start passStruct1
   :end-before: end passStruct1

:file:`struct.yaml`:

.. code-block:: yaml

    - decl: int passStruct1(Cstruct1 *s1)

Fortran calls C via the following interface:

.. literalinclude:: ../regression/reference/struct/wrapfstruct.f
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

C library function:

.. literalinclude:: ../regression/run/struct/struct.c
   :language: c
   :start-after: start passStructByValue
   :end-before: end passStructByValue

:file:`struct.yaml`:

.. code-block:: yaml

    - decl: double passStructByValue(struct1 arg)

Fortran calls C via the following interface:

.. literalinclude:: ../regression/reference/struct/wrapfstruct.f
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

The C++ header file from :file:`tutorial.hpp`.

.. code-block:: c++

    class Class1
    {
    public:
        int m_flag;
        int m_test;
        Class1()         : m_flag(0), m_test(0)    {};
        Class1(int flag) : m_flag(flag), m_test(0) {};
    };

:file:`tutorial.yaml`:

.. code-block:: yaml

    declarations:
    - decl: class Class1
      declarations:
      - decl: Class1()         +name(new)
        format:
          function_suffix: _default
      - decl: Class1(int flag) +name(new)
        format:
        function_suffix: _flag
      - decl: ~Class1() +name(delete)

A C wrapper function is created for each constructor and the destructor.

The C wrappers:

.. literalinclude:: ../regression/reference/tutorial/wrapClass1.cpp
   :language: c
   :start-after: start TUT_class1_new_default
   :end-before: end TUT_class1_new_default

.. literalinclude:: ../regression/reference/tutorial/wrapClass1.cpp
   :language: c
   :start-after: start TUT_class1_new_flag
   :end-before: end TUT_class1_new_flag

.. literalinclude:: ../regression/reference/tutorial/wrapClass1.cpp
   :language: c
   :start-after: start TUT_class1_delete
   :end-before: end TUT_class1_delete

The corresponding Fortran interfaces:

.. literalinclude:: ../regression/reference/tutorial/wrapftutorial.f
   :language: fortran
   :start-after: start c_class1_new_default
   :end-before: end c_class1_new_default
   :dedent: 4

.. literalinclude:: ../regression/reference/tutorial/wrapftutorial.f
   :language: fortran
   :start-after: start c_class1_new_flag
   :end-before: end c_class1_new_flag
   :dedent: 4

.. literalinclude:: ../regression/reference/tutorial/wrapftutorial.f
   :language: fortran
   :start-after: start c_class1_delete
   :end-before: end c_class1_delete
   :dedent: 4

And the Fortran wrappers:

.. literalinclude:: ../regression/reference/tutorial/wrapftutorial.f
   :language: fortran
   :start-after: start class1_new_default
   :end-before: end class1_new_default
   :dedent: 4

.. literalinclude:: ../regression/reference/tutorial/wrapftutorial.f
   :language: fortran
   :start-after: start class1_new_flag
   :end-before: end class1_new_flag
   :dedent: 4

.. literalinclude:: ../regression/reference/tutorial/wrapftutorial.f
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

.. code-block:: fortran

    interface class1_new
        module procedure class1_new_default
        module procedure class1_new_flag
    end interface class1_new

A class instance is created and destroy from Fortran as:

.. code-block:: fortran

    use tutorial_mod
    type(class1) obj

    obj = class1_new()
    call obj%delete

Corresponding C++ code:

.. code-block:: c++

    include <tutorial.hpp>

    tutorial::Class1 * obj = new tutorial::Class1;

    delete obj;

.. ############################################################

.. _example_getter_and_setter:

Getter and Setter
^^^^^^^^^^^^^^^^^
The C++ header file from :file:`tutorial.hpp`.

.. code-block:: c++

    class Class1
    {
    public:
        int m_flag;
        int m_test;
    };

:file:`tutorial.yaml`:

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

.. literalinclude:: ../regression/reference/tutorial/wrapClass1.cpp
   :language: c
   :start-after: start TUT_class1_get_m_flag
   :end-before: end TUT_class1_get_m_flag

.. literalinclude:: ../regression/reference/tutorial/wrapClass1.cpp
   :language: c
   :start-after: start TUT_class1_get_test
   :end-before: end TUT_class1_get_test

.. literalinclude:: ../regression/reference/tutorial/wrapClass1.cpp
   :language: c
   :start-after: start TUT_class1_set_test
   :end-before: end TUT_class1_set_test

The corresponding Fortran interfaces:

.. literalinclude:: ../regression/reference/tutorial/wrapftutorial.f
   :language: fortran
   :start-after: start c_class1_get_m_flag
   :end-before: end c_class1_get_m_flag
   :dedent: 4

.. literalinclude:: ../regression/reference/tutorial/wrapftutorial.f
   :language: fortran
   :start-after: start c_class1_get_test
   :end-before: end c_class1_get_test
   :dedent: 4

.. literalinclude:: ../regression/reference/tutorial/wrapftutorial.f
   :language: fortran
   :start-after: start c_class1_set_test
   :end-before: end c_class1_set_test
   :dedent: 4

And the Fortran wrappers:

.. literalinclude:: ../regression/reference/tutorial/wrapftutorial.f
   :language: fortran
   :start-after: start class1_get_m_flag
   :end-before: end class1_get_m_flag
   :dedent: 4

.. literalinclude:: ../regression/reference/tutorial/wrapftutorial.f
   :language: fortran
   :start-after: start class1_get_test
   :end-before: end class1_get_test
   :dedent: 4

.. literalinclude:: ../regression/reference/tutorial/wrapftutorial.f
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

    use tutorial_mod
    type(class1) obj
    integer iflag

    obj = class1_new()
    call obj%set_test(4)
    iflag = obj%get_test()

Corresponding C++ code:

.. code-block:: c++

    include <tutorial.hpp>
    tutorial::Class1 obj = new * tutorial::Class1;
    obj->m_test = 4;
    int iflag = obj->m_test;

.. ############################################################

.. _example_UseDefaultArguments:

Default Value Arguments
-----------------------

The default values are provided in the function declaration.

:file:`tutorial.hpp`:

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
   :start-after: start TUT_use_default_arguments
   :end-before: end TUT_use_default_arguments

.. literalinclude:: ../regression/reference/tutorial/wrapTutorial.cpp
   :language: c++
   :start-after: start TUT_use_default_arguments_arg1
   :end-before: end TUT_use_default_arguments_arg1

.. literalinclude:: ../regression/reference/tutorial/wrapTutorial.cpp
   :language: c++
   :start-after: start TUT_use_default_arguments_arg1_arg2
   :end-before: end TUT_use_default_arguments_arg1_arg2

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
