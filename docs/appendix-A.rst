
Sample Fortran Wrappers
=======================

This chapter gives details of the generated code.
It's intended for users who want to understand the details
of how the wrappers are created.

All of these examples are derived from tests in the ``regression``
directory.

Numeric Types
-------------

other
^^^^^
.. ############################################################

.. _example_PassByValue:

PassByValue
"""""""""""

YAML:

.. code-block:: yaml

    - decl: double PassByValue(double arg1, int arg2)

Both types are supported directly by the ``iso_c_binding`` module
so there is no need for a Fortran function.
The C function can be called directly by Fortran
by using the ``bind(C)`` keyword.

Calls C via the interface:

.. literalinclude:: ../regression/reference/clibrary/wrapfclibrary.f
   :language: fortran
   :start-after: start pass_by_value
   :end-before: end pass_by_value
   :dedent: 8

C library function:

.. literalinclude:: ../regression/run/clibrary/clibrary.c
   :language: c
   :start-after: start PassByValue
   :end-before: end PassByValue

.. ############################################################

.. _example_PassByReference:

PassByReference
"""""""""""""""

YAML:

.. code-block:: yaml

    - decl: void PassByReference(double *arg1+intent(in), int *arg2+intent(out))

Calls C via the interface:

.. literalinclude:: ../regression/reference/clibrary/wrapfclibrary.f
   :language: fortran
   :start-after: start pass_by_reference
   :end-before: end pass_by_reference
   :dedent: 8

C library function:

.. literalinclude:: ../regression/run/clibrary/clibrary.c
   :language: c
   :start-after: start PassByReference
   :end-before: end PassByReference

Example usage:

.. code-block:: fortran

    integer(C_INT) var
    call pass_by_reference(3.14d0, var)
    call assert_equals(3, var)

.. ############################################################

.. _example_Sum:

Sum
"""

YAML:

.. code-block:: yaml

   - decl: void Sum(int len +implied(size(values)),
                    int *values +dimension(:)+intent(in),
                    int *result +intent(out))

The Fortran wrapper:

.. literalinclude:: ../regression/reference/clibrary/wrapfclibrary.f
   :language: fortran
   :start-after: start sum
   :end-before: end sum
   :dedent: 4

Calls C via the interface:

.. literalinclude:: ../regression/reference/clibrary/wrapfclibrary.f
   :language: fortran
   :start-after: start c_sum
   :end-before: end c_sum
   :dedent: 8

C library function:

.. literalinclude:: ../regression/run/clibrary/clibrary.c
   :language: c
   :start-after: start Sum
   :end-before: end Sum

.. _example_getMinMax:

.. ############################################################

getMinMax
"""""""""

No Fortran function is created.  Only an interface to a C wrapper
which dereference the pointers so they can be treated as references.

YAML:

.. code-block:: yaml

    - decl: void getMinMax(int &min +intent(out), int &max +intent(out))

Calls C via the interface:

.. literalinclude:: ../regression/reference/tutorial/wrapftutorial.f
   :language: fortran
   :start-after: start get_min_max
   :end-before: end get_min_max
   :dedent: 8

The C wrapper:

.. literalinclude:: ../regression/reference/tutorial/wrapTutorial.cpp
   :language: c
   :start-after: start TUT_get_min_max
   :end-before: end TUT_get_min_max

C++ library function:

.. literalinclude:: ../regression/run/tutorial/tutorial.cpp
   :language: c
   :start-after: start getMinMax
   :end-before: end getMinMax


Bool
----

other
^^^^^
.. ############################################################

.. _example_checkBool:

checkBool
"""""""""

Assignments are done to convert between ``logical`` and
``logical(C_BOOL)``.

YAML:

.. code-block:: yaml

    - decl: void checkBool(const bool arg1,
                           bool *arg2+intent(out),
                           bool *arg3+intent(inout))

The Fortran wrapper:

.. literalinclude:: ../regression/reference/clibrary/wrapfclibrary.f
   :language: fortran
   :start-after: start check_bool
   :end-before: end check_bool
   :dedent: 4

Calls C via the interface:

.. literalinclude:: ../regression/reference/clibrary/wrapfclibrary.f
   :language: fortran
   :start-after: start c_check_bool
   :end-before: end c_check_bool
   :dedent: 8


Character
---------


Char
^^^^

.. ############################################################

.. _example_acceptName:

acceptName
""""""""""

Pass a ``NULL`` terminated string to a C function.
The string will be unchanged.

YAML:

.. code-block:: yaml

  - decl: void acceptName(const char *name)

The Fortran wrapper:

.. literalinclude:: ../regression/reference/clibrary/wrapfclibrary.f
   :language: fortran
   :start-after: start accept_name
   :end-before: end accept_name
   :dedent: 4

Calls C via the interface:

.. literalinclude:: ../regression/reference/clibrary/wrapfclibrary.f
   :language: fortran
   :start-after: start c_accept_name
   :end-before: end c_accept_name
   :dedent: 8

.. literalinclude:: ../regression/reference/clibrary/wrapClibrary.c
   :language: c
   :start-after: start CLI_accept_name_bufferify
   :end-before: end CLI_accept_name_bufferify

.. ############################################################

.. _example_returnOneName:

returnOneName
"""""""""""""

Pass the pointer to a buffer which the C library will fill.
The length of the string is unknown.

YAML:

.. code-block:: yaml

    - decl: void returnOneName(char *name1+intent(out)+charlen(MAXNAME))

The Fortran wrapper:

.. literalinclude:: ../regression/reference/clibrary/wrapfclibrary.f
   :language: fortran
   :start-after: start return_one_name
   :end-before: end return_one_name
   :dedent: 4

Calls C via the interface:

.. literalinclude:: ../regression/reference/clibrary/wrapfclibrary.f
   :language: fortran
   :start-after: start c_return_one_name_bufferify
   :end-before: end c_return_one_name_bufferify
   :dedent: 8

The C wrapper:

.. literalinclude:: ../regression/reference/clibrary/wrapClibrary.c
   :language: c
   :start-after: start CLI_return_one_name_bufferify
   :end-before: end CLI_return_one_name_bufferify

.. ############################################################

.. _example_passCharPtr:

passCharPtr
"""""""""""

The function ``passCharPtr(dest, src)`` is equivalent to the Fortran
statement ``dest = src``:


YAML:

.. code-block:: yaml

    - decl: void passCharPtr(char * dest+intent(out)+charlen(40),
                             const char *src)

The intent of ``dest`` must be explicit.  It defaults to *intent(inout)*
since it is a pointer.
``src`` is implied to be *intent(in)* since it is ``const``.
This single line will create five different wrappers.

The Fortran wrapper:

.. literalinclude:: ../regression/reference/strings/wrapfstrings.f
   :language: fortran
   :start-after: start pass_char_ptr
   :end-before: end pass_char_ptr
   :dedent: 4

Calls C via the interface:

.. literalinclude:: ../regression/reference/strings/wrapfstrings.f
   :language: fortran
   :start-after: start c_pass_char_ptr
   :end-before: end c_pass_char_ptr
   :dedent: 8

.. literalinclude:: ../regression/reference/strings/wrapfstrings.f
   :language: fortran
   :start-after: start c_pass_char_ptr_bufferify
   :end-before: end c_pass_char_ptr_bufferify
   :dedent: 8

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

C library function:

.. literalinclude:: ../regression/run/strings/strings.cpp
   :language: c
   :start-after: start passCharPtr
   :end-before: end passCharPtr

The function can be called without the user aware that it is written in C++:

.. code-block:: fortran

    character(30) str
    call pass_char_ptr(dest=str, src="mouse")


.. ############################################################

.. _example_ImpliedTextLen:

ImpliedTextLen
""""""""""""""

Pass the pointer to a buffer which the C library will fill.  The
length of the buffer is passed in ``ltext``.  Since Fortran knows the
length of ``CHARACTER`` variable, the Fortran wrapper does not need to
be explicitly told the length of the variable.  Instead its value can
be defined with the *implied* attribute.

This can be used to emulate the behavior of most Fortran compilers
which will pass an additional, hidden argument which contains the
length of a ``CHARACTER`` argument.

YAML:

.. code-block:: yaml

    - decl: void ImpliedTextLen(char *text+intent(out)+charlen(MAXNAME),
                                int ltext+implied(len(text)))

The Fortran wrapper:

.. literalinclude:: ../regression/reference/clibrary/wrapfclibrary.f
   :language: fortran
   :start-after: start implied_text_len
   :end-before: end implied_text_len
   :dedent: 4

Calls C via the interface:

.. literalinclude:: ../regression/reference/clibrary/wrapfclibrary.f
   :language: fortran
   :start-after: start c_implied_text_len_bufferify
   :end-before: end c_implied_text_len_bufferify
   :dedent: 8

The C wrapper:

.. literalinclude:: ../regression/reference/clibrary/wrapClibrary.c
   :language: c
   :start-after: start CLI_implied_text_len_bufferify
   :end-before: end CLI_implied_text_len_bufferify


std::string
^^^^^^^^^^^

.. ############################################################

.. _example_acceptStringReference:

acceptStringReference
"""""""""""""""""""""

YAML:

.. code-block:: yaml

    - decl: void acceptStringReference(std::string & arg1)

The Fortran wrapper:

.. literalinclude:: ../regression/reference/strings/wrapfstrings.f
   :language: fortran
   :start-after: start accept_string_reference
   :end-before: end accept_string_reference
   :dedent: 4

Calls C via the interface:

.. literalinclude:: ../regression/reference/strings/wrapfstrings.f
   :language: fortran
   :start-after: start c_accept_string_reference_bufferify
   :end-before: end c_accept_string_reference_bufferify
   :dedent: 8

The C wrapper:

.. literalinclude:: ../regression/reference/strings/wrapstrings.cpp
   :language: c
   :start-after: start STR_accept_string_reference_bufferify
   :end-before: end STR_accept_string_reference_bufferify

char functions
^^^^^^^^^^^^^^

.. ############################################################

.. _example_getCharPtr1:

getCharPtr1
"""""""""""

Return a pointer and convert into an ``ALLOCATABLE`` ``CHARACTER``
variable.  The Fortran application is responsible to release the
memory.  However, this may be done automatically by the Fortran
runtime.

YAML:

.. code-block:: yaml

    - decl: const char * getCharPtr1()

The Fortran wrapper uses the metadata in ``DSHF_rv`` to allocate
a ``CHARACTER`` variable of the correct length.
The helper function ``SHROUD_copy_string_and_free`` will copy 
the results of the C++ function into the return variable:

.. literalinclude:: ../regression/reference/strings/wrapfstrings.f
   :language: fortran
   :start-after: start get_char_ptr1
   :end-before: end get_char_ptr1
   :dedent: 4

Calls C via the interface:

.. literalinclude:: ../regression/reference/strings/wrapfstrings.f
   :language: fortran
   :start-after: start c_get_char_ptr1_bufferify
   :end-before: end c_get_char_ptr1_bufferify
   :dedent: 8

The C wrapper copies all of the metadata into a ``SHROUD_array``
struct which is used by the Fortran wrapper:

.. literalinclude:: ../regression/reference/strings/wrapstrings.cpp
   :language: c
   :start-after: start STR_get_char_ptr1_bufferify
   :end-before: end STR_get_char_ptr1_bufferify

Fortran usage:

.. code-block:: fortran

    character(len=:), allocatable :: str
    str = get_char_ptr1()

.. ############################################################

.. _example_getCharPtr2:

getCharPtr2
"""""""""""

If you know the maximum size of string that you expect the function to
return, then the *len* attribute is used to declare the length.  The
explicit ``ALLOCATE`` is avoided but any result which is longer than
the length will be silently truncated.

YAML:

.. code-block:: yaml

    - decl: const char * getCharPtr2() +len(30)

The Fortran wrapper:

.. literalinclude:: ../regression/reference/strings/wrapfstrings.f
   :language: fortran
   :start-after: start get_char_ptr2
   :end-before: end get_char_ptr2
   :dedent: 4

Calls C via the interface:

.. literalinclude:: ../regression/reference/strings/wrapfstrings.f
   :language: fortran
   :start-after: start c_get_char_ptr2_bufferify
   :end-before: end c_get_char_ptr2_bufferify
   :dedent: 8

The C wrapper:

.. literalinclude:: ../regression/reference/strings/wrapstrings.cpp
   :language: c
   :start-after: start STR_get_char_ptr2_bufferify
   :end-before: end STR_get_char_ptr2_bufferify

Fortran usage:

.. code-block:: fortran

    character(30) str
    str = get_char_ptr2()

.. ############################################################

.. _example_getCharPtr3:

getCharPtr3
"""""""""""

Create a Fortran subroutine in an additional ``CHARACTER``
argument for the C function result. Any size character string can
be returned limited by the size of the Fortran argument.  The
argument is defined by the *F_string_result_as_arg* format string.
Works with Fortran 90.

YAML:

.. code-block:: yaml

    - decl: const char * getCharPtr3()
      format:
        F_string_result_as_arg: output

The Fortran wrapper:

.. literalinclude:: ../regression/reference/strings/wrapfstrings.f
   :language: fortran
   :start-after: start get_char_ptr3
   :end-before: end get_char_ptr3
   :dedent: 4

Calls C via the interface:

.. literalinclude:: ../regression/reference/strings/wrapfstrings.f
   :language: fortran
   :start-after: start c_get_char_ptr3_bufferify
   :end-before: end c_get_char_ptr3_bufferify
   :dedent: 8

The C wrapper:

.. literalinclude:: ../regression/reference/strings/wrapstrings.cpp
   :language: c
   :start-after: start STR_get_char_ptr3_bufferify
   :end-before: end STR_get_char_ptr3_bufferify

Fortran usage:

.. code-block:: fortran

    character(30) str
    call get_char_ptrs(str)

string functions
^^^^^^^^^^^^^^^^

.. ############################################################

.. _example_getConstStringRefPure:

getConstStringRefPure
"""""""""""""""""""""

YAML:

.. code-block:: yaml

    - decl: const string& getConstStringRefPure()

The Fortran wrapper:

.. literalinclude:: ../regression/reference/strings/wrapfstrings.f
   :language: fortran
   :start-after: start get_const_string_ref_pure
   :end-before: end get_const_string_ref_pure
   :dedent: 4

Calls C via the interface:

.. literalinclude:: ../regression/reference/strings/wrapfstrings.f
   :language: fortran
   :start-after: start c_get_const_string_ref_pure_bufferify
   :end-before: end c_get_const_string_ref_pure_bufferify
   :dedent: 8

The C wrapper:

.. literalinclude:: ../regression/reference/strings/wrapstrings.cpp
   :language: c
   :start-after: start STR_get_const_string_ref_pure_bufferify
   :end-before: end STR_get_const_string_ref_pure_bufferify

std::vector
-----------

other
^^^^^

.. ############################################################

.. _example_vector_sum:

vector_sum
""""""""""

YAML:

.. code-block:: yaml

    - decl: int vector_sum(const std::vector<int> &arg)

The Fortran wrapper:

.. literalinclude:: ../regression/reference/vectors/wrapfvectors.f
   :language: fortran
   :start-after: start vector_sum
   :end-before: end vector_sum
   :dedent: 4

Calls C via the interface:

.. literalinclude:: ../regression/reference/vectors/wrapfvectors.f
   :language: fortran
   :start-after: start c_vector_sum_bufferify
   :end-before: end c_vector_sum_bufferify
   :dedent: 8

The C wrapper:

.. literalinclude:: ../regression/reference/vectors/wrapvectors.cpp
   :language: c
   :start-after: start VEC_vector_sum_bufferify
   :end-before: end VEC_vector_sum_bufferify

.. ############################################################

.. _example_vector_iota_out:

vector_iota_out
"""""""""""""""

YAML:

.. code-block:: yaml

    - decl: void vector_iota_out(std::vector<int> &arg+intent(out))

The Fortran wrapper:

.. literalinclude:: ../regression/reference/vectors/wrapfvectors.f
   :language: fortran
   :start-after: start vector_iota_out
   :end-before: end vector_iota_out
   :dedent: 4

Calls C via the interface:

.. literalinclude:: ../regression/reference/vectors/wrapfvectors.f
   :language: fortran
   :start-after: start c_vector_iota_out_bufferify
   :end-before: end c_vector_iota_out_bufferify
   :dedent: 8

The C wrapper:

.. literalinclude:: ../regression/reference/vectors/wrapvectors.cpp
   :language: c
   :start-after: start VEC_vector_iota_out_bufferify
   :end-before: end VEC_vector_iota_out_bufferify

.. ############################################################

.. _example_vector_iota_out_alloc:

vector_iota_out_alloc
"""""""""""""""""""""

YAML:

.. code-block:: yaml

    - decl: void vector_iota_out_alloc(std::vector<int> &arg+intent(out)+deref(allocatable))

The Fortran wrapper:

.. literalinclude:: ../regression/reference/vectors/wrapfvectors.f
   :language: fortran
   :start-after: start vector_iota_out_alloc
   :end-before: end vector_iota_out_alloc
   :dedent: 4

Calls C via the interface:

.. literalinclude:: ../regression/reference/vectors/wrapfvectors.f
   :language: fortran
   :start-after: start c_vector_iota_out_alloc_bufferify
   :end-before: end c_vector_iota_out_alloc_bufferify
   :dedent: 8

The C wrapper:

.. literalinclude:: ../regression/reference/vectors/wrapvectors.cpp
   :language: c
   :start-after: start VEC_vector_iota_out_alloc_bufferify
   :end-before: end VEC_vector_iota_out_alloc_bufferify

.. ############################################################

.. _example_vector_iota_inout_alloc:

vector_iota_inout_alloc
"""""""""""""""""""""""

YAML:

.. code-block:: yaml

    - decl: void vector_iota_out_alloc(std::vector<int> &arg+intent(inout)+deref(allocatable))

The Fortran wrapper:

.. literalinclude:: ../regression/reference/vectors/wrapfvectors.f
   :language: fortran
   :start-after: start vector_iota_inout_alloc
   :end-before: end vector_iota_inout_alloc
   :dedent: 4

Calls C via the interface:

.. literalinclude:: ../regression/reference/vectors/wrapfvectors.f
   :language: fortran
   :start-after: start c_vector_iota_inout_alloc_bufferify
   :end-before: end c_vector_iota_inout_alloc_bufferify
   :dedent: 8

The C wrapper:

.. literalinclude:: ../regression/reference/vectors/wrapvectors.cpp
   :language: c
   :start-after: start VEC_vector_iota_inout_alloc_bufferify
   :end-before: end VEC_vector_iota_inout_alloc_bufferify




Class Type
----------

