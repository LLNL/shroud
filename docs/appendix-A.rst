
Sample output
==============

This chapter gives details of the generated code.
It's intended for users who want to understand the details
of how the wrappers are created.

All of these examples are derived from tests in the ``regression``
directory.

Bool
----

other
^^^^^

.. _example_checkBool:

checkBool
"""""""""

YAML:

.. code-block:: yaml

    - decl: void checkBool(const bool arg1,
                           bool *arg2+intent(out),
                           bool *arg3+intent(inout))

The Fortran wrapper:

.. literalinclude:: ../regression/reference/clibrary/wrapfclibrary.f
   :language: fortran
   :start-after: before check_bool
   :end-before: after check_bool
   :dedent: 4

Calls C via the interface:

.. literalinclude:: ../regression/reference/clibrary/wrapfclibrary.f
   :language: fortran
   :start-after: before c_check_bool
   :end-before: after c_check_bool
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
   :start-after: before accept_name
   :end-before: after accept_name
   :dedent: 4

Calls C via the interface:

.. literalinclude:: ../regression/reference/clibrary/wrapfclibrary.f
   :language: fortran
   :start-after: before c_accept_name
   :end-before: after c_accept_name
   :dedent: 8

.. literalinclude:: ../regression/reference/clibrary/wrapClibrary.c
   :language: c
   :start-after: before CLI_accept_name_bufferify
   :end-before: after CLI_accept_name_bufferify

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
   :start-after: before return_one_name
   :end-before: after return_one_name
   :dedent: 4

Calls C via the interface:

.. literalinclude:: ../regression/reference/clibrary/wrapfclibrary.f
   :language: fortran
   :start-after: before c_return_one_name_bufferify
   :end-before: after c_return_one_name_bufferify
   :dedent: 8

The C wrapper:

.. literalinclude:: ../regression/reference/clibrary/wrapClibrary.c
   :language: c
   :start-after: before CLI_return_one_name_bufferify
   :end-before: after CLI_return_one_name_bufferify

.. _example_ImpliedTextLen:

ImpliedTextLen
""""""""""""""

Pass the pointer to a buffer which the C library will fill.
The length of the buffer is ``ltext``.

YAML:

.. code-block:: yaml

    - decl: void ImpliedTextLen(char *text+intent(out)+charlen(MAXNAME),
                                int ltext+implied(len(text)))

The Fortran wrapper:

.. literalinclude:: ../regression/reference/clibrary/wrapfclibrary.f
   :language: fortran
   :start-after: before implied_text_len
   :end-before: after implied_text_len
   :dedent: 4

Calls C via the interface:

.. literalinclude:: ../regression/reference/clibrary/wrapfclibrary.f
   :language: fortran
   :start-after: before c_implied_text_len_bufferify
   :end-before: after c_implied_text_len_bufferify
   :dedent: 8

The C wrapper:

.. literalinclude:: ../regression/reference/clibrary/wrapClibrary.c
   :language: c
   :start-after: before CLI_implied_text_len_bufferify
   :end-before: after CLI_implied_text_len_bufferify

.. ############################################################

std::string
^^^^^^^^^^^

.. _example_acceptStringReference:

acceptStringReference
"""""""""""""""""""""

YAML:

.. code-block:: yaml

    - decl: void acceptStringReference(std::string & arg1)

The Fortran wrapper:

.. literalinclude:: ../regression/reference/strings/wrapfstrings.f
   :language: fortran
   :start-after: before accept_string_reference
   :end-before: after accept_string_reference
   :dedent: 4

Calls C via the interface:

.. literalinclude:: ../regression/reference/strings/wrapfstrings.f
   :language: fortran
   :start-after: before c_accept_string_reference_bufferify
   :end-before: after c_accept_string_reference_bufferify
   :dedent: 8

The C wrapper:

.. literalinclude:: ../regression/reference/strings/wrapstrings.cpp
   :language: c
   :start-after: before STR_accept_string_reference_bufferify
   :end-before: after STR_accept_string_reference_bufferify

.. ############################################################

char functions
^^^^^^^^^^^^^^

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

The Fortran wrapper:

.. literalinclude:: ../regression/reference/strings/wrapfstrings.f
   :language: fortran
   :start-after: before get_char_ptr1
   :end-before: after get_char_ptr1
   :dedent: 4

Calls C via the interface:

.. literalinclude:: ../regression/reference/strings/wrapfstrings.f
   :language: fortran
   :start-after: before c_get_char_ptr1_bufferify
   :end-before: after c_get_char_ptr1_bufferify
   :dedent: 8

The C wrapper:

.. literalinclude:: ../regression/reference/strings/wrapstrings.cpp
   :language: c
   :start-after: before STR_get_char_ptr1_bufferify
   :end-before: after STR_get_char_ptr1_bufferify

.. _example_getCharPtr2:

getCharPtr2
"""""""""""

Create a Fortran function which returns a predefined ``CHARACTER`` 
value.  The size is determined by the *len* argument on the function.
This is useful when the maximum size is already known.
Works with Fortran 90.

YAML:

.. code-block:: yaml

    - decl: const char * getCharPtr2() +len(30)

The Fortran wrapper:

.. literalinclude:: ../regression/reference/strings/wrapfstrings.f
   :language: fortran
   :start-after: before get_char_ptr2
   :end-before: after get_char_ptr2
   :dedent: 4

Calls C via the interface:

.. literalinclude:: ../regression/reference/strings/wrapfstrings.f
   :language: fortran
   :start-after: before c_get_char_ptr2_bufferify
   :end-before: after c_get_char_ptr2_bufferify
   :dedent: 8

The C wrapper:

.. literalinclude:: ../regression/reference/strings/wrapstrings.cpp
   :language: c
   :start-after: before STR_get_char_ptr2_bufferify
   :end-before: after STR_get_char_ptr2_bufferify

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
   :start-after: before get_char_ptr3
   :end-before: after get_char_ptr3
   :dedent: 4

Calls C via the interface:

.. literalinclude:: ../regression/reference/strings/wrapfstrings.f
   :language: fortran
   :start-after: before c_get_char_ptr3_bufferify
   :end-before: after c_get_char_ptr3_bufferify
   :dedent: 8

The C wrapper:

.. literalinclude:: ../regression/reference/strings/wrapstrings.cpp
   :language: c
   :start-after: before STR_get_char_ptr3_bufferify
   :end-before: after STR_get_char_ptr3_bufferify

string functions
^^^^^^^^^^^^^^^^

.. _example_getConstStringRefPure:

getConstStringRefPure
"""""""""""""""""""""

YAML:

.. code-block:: yaml

    - decl: const string& getConstStringRefPure()

The Fortran wrapper:

.. literalinclude:: ../regression/reference/strings/wrapfstrings.f
   :language: fortran
   :start-after: before get_const_string_ref_pure
   :end-before: after get_const_string_ref_pure
   :dedent: 4

Calls C via the interface:

.. literalinclude:: ../regression/reference/strings/wrapfstrings.f
   :language: fortran
   :start-after: before c_get_const_string_ref_pure_bufferify
   :end-before: after c_get_const_string_ref_pure_bufferify
   :dedent: 8

The C wrapper:

.. literalinclude:: ../regression/reference/strings/wrapstrings.cpp
   :language: c
   :start-after: before STR_get_const_string_ref_pure_bufferify
   :end-before: after STR_get_const_string_ref_pure_bufferify

std::vector
-----------

other
^^^^^

.. _example_vector_sum:

vector_sum
""""""""""

YAML:

.. code-block:: yaml

    - decl: int vector_sum(const std::vector<int> &arg)

The Fortran wrapper:

.. literalinclude:: ../regression/reference/vectors/wrapfvectors.f
   :language: fortran
   :start-after: before vector_sum
   :end-before: after vector_sum
   :dedent: 4

Calls C via the interface:

.. literalinclude:: ../regression/reference/vectors/wrapfvectors.f
   :language: fortran
   :start-after: before c_vector_sum_bufferify
   :end-before: after c_vector_sum_bufferify
   :dedent: 8

The C wrapper:

.. literalinclude:: ../regression/reference/vectors/wrapvectors.cpp
   :language: c
   :start-after: before VEC_vector_sum_bufferify
   :end-before: after VEC_vector_sum_bufferify

.. _example_vector_iota_out:

vector_iota_out
"""""""""""""""

YAML:

.. code-block:: yaml

    - decl: void vector_iota_out(std::vector<int> &arg+intent(out))

The Fortran wrapper:

.. literalinclude:: ../regression/reference/vectors/wrapfvectors.f
   :language: fortran
   :start-after: before vector_iota_out
   :end-before: after vector_iota_out
   :dedent: 4

Calls C via the interface:

.. literalinclude:: ../regression/reference/vectors/wrapfvectors.f
   :language: fortran
   :start-after: before c_vector_iota_out_bufferify
   :end-before: after c_vector_iota_out_bufferify
   :dedent: 8

The C wrapper:

.. literalinclude:: ../regression/reference/vectors/wrapvectors.cpp
   :language: c
   :start-after: before VEC_vector_iota_out_bufferify
   :end-before: after VEC_vector_iota_out_bufferify


.. _example_vector_iota_out_alloc:

vector_iota_out_alloc
"""""""""""""""""""""

YAML:

.. code-block:: yaml

    - decl: void vector_iota_out_alloc(std::vector<int> &arg+intent(out)+deref(allocatable))

The Fortran wrapper:

.. literalinclude:: ../regression/reference/vectors/wrapfvectors.f
   :language: fortran
   :start-after: before vector_iota_out_alloc
   :end-before: after vector_iota_out_alloc
   :dedent: 4

Calls C via the interface:

.. literalinclude:: ../regression/reference/vectors/wrapfvectors.f
   :language: fortran
   :start-after: before c_vector_iota_out_alloc_bufferify
   :end-before: after c_vector_iota_out_alloc_bufferify
   :dedent: 8

The C wrapper:

.. literalinclude:: ../regression/reference/vectors/wrapvectors.cpp
   :language: c
   :start-after: before VEC_vector_iota_out_alloc_bufferify
   :end-before: after VEC_vector_iota_out_alloc_bufferify

.. _example_vector_iota_inout_alloc:

vector_iota_inout_alloc
"""""""""""""""""""""""

YAML:

.. code-block:: yaml

    - decl: void vector_iota_out_alloc(std::vector<int> &arg+intent(inout)+deref(allocatable))

The Fortran wrapper:

.. literalinclude:: ../regression/reference/vectors/wrapfvectors.f
   :language: fortran
   :start-after: before vector_iota_inout_alloc
   :end-before: after vector_iota_inout_alloc
   :dedent: 4

Calls C via the interface:

.. literalinclude:: ../regression/reference/vectors/wrapfvectors.f
   :language: fortran
   :start-after: before c_vector_iota_inout_alloc_bufferify
   :end-before: after c_vector_iota_inout_alloc_bufferify
   :dedent: 8

The C wrapper:

.. literalinclude:: ../regression/reference/vectors/wrapvectors.cpp
   :language: c
   :start-after: before VEC_vector_iota_inout_alloc_bufferify
   :end-before: after VEC_vector_iota_inout_alloc_bufferify




Class Type
----------

