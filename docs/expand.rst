.. Copyright Shroud Project Developers. See LICENSE file for details.

   SPDX-License-Identifier: (BSD-3-Clause)

How to Expand Functionality
===========================

The wrapping features of Shroud are controlled by data files which are
read in upon startup.  This allows Shroud to wrap many types of
arguments without any additional input from the user.  However, there
will always be situations that require some additional ability.  This
section describes the input data files and how a user may add to them.

Shroud can be thought of as a fancy macro processor.  It reads the
input and performs lots of redundant, tedious replacements. One
function argument causes several layers of code to be generated which
involve transformations of the input. To help manage the redundency,
several layers of abstraction are provided.

The lowest layer is the :ref:`typemap <TypemapsAnchor>`.  It factors
out some common fields for individual types.  For example, most
'native' types such as ``int`` and ``double`` will produce identical
statements except for things such as Fortran ``kind`` (``C_INT``,
``C_DOUBLE``), intrinsic functions (``int``, ``real``) and C types
(``int``, ``double``).

The next layer is the format dictionary.  This is created for each
argument for each language. Some values are computed from the argument
description and attribute. Other values can be added directly by the
user in the YAML file.

Finally, :ref:`statements <StatementsAnchor>` use format entries to
generate code at the many locations required to create the wrappers.


Tutorial
--------

This tutorial will work through how a statement group is developed
for a Fortran wrapper.
The function ``fetchCharPtr`` returns a pointer to a ``char`` array
in one of the arguments.

.. code-block:: yaml

    options:
      debug: True
      wrap_c: False

    declarations:
    - decl: void fetchCharPtr(char **fetch1+intent(out))

Note the use of the **debug** option.
This will write additional comments into the generated code that will list
the statement groups used for each argument and the function result.
Running :code:`Shroud` will initially report that the statment group is not found.

.. code-block:: text

    Phase: FillMeta function
    ----------------------------------------
    Node: fetchCharPtr
    line 15
    Unknown statement: f_out_char**_cdesc_pointer

The first step is to define the statement group by name.
Since this is for a builtin statement group, it is added to
``shroud/fc-statements.json``.

The group name is derived from the language, intent, type and
defaults for the *+api* and *+deref* attributes.

The *usage* and *notes* sections are optional, but will help
document the intended usage of the group.

.. code-block:: json

    {
        "name": "f_out_char**_cdesc_pointer",
        "usage": [
            "char **arg +intent(out)"
        ],
        "notes": [
            "Return a Fortran pointer to C memory."
        ]
    }

:code:`Shroud`
will now generate wrapper in Three parts. The interface, the Fortran
wrapper and the C wrapper.

.. code-block:: fortran

    interface
        ! ----------------------------------------
        ! Function:  void fetchCharPtr
        ! Statement: f_subroutine
        ! ----------------------------------------
        ! Argument:  char **fetch1 +intent(out)
        ! Statement: f_out_char**_cdesc_pointer
        subroutine fetch_char_ptr() &
                bind(C, name="TES_fetchCharPtr_bufferify")
            implicit none
        end subroutine fetch_char_ptr
    end interface

Since there are no code required in the Fortran wrapper, only the
interface is necessary. The **debug** option will show how the Fortran
wrapper would be created, but it is conditionally compiled out.
    
.. code-block:: fortran

    #if 0
    ! Only the interface is needed
    ! ----------------------------------------
    ! Function:  void fetchCharPtr
    ! Statement: f_subroutine
    ! ----------------------------------------
    ! Argument:  char **fetch1 +intent(out)
    ! Statement: f_out_char**_cdesc_pointer
    subroutine fetch_char_ptr()
        ! splicer begin function.fetch_char_ptr
        call c_fetch_char_ptr_bufferify(fetch1)
        ! splicer end function.fetch_char_ptr
    end subroutine fetch_char_ptr
    #endif

.. code-block:: c++
                
    // ----------------------------------------
    // Function:  void fetchCharPtr
    // Statement: f_subroutine
    // ----------------------------------------
    // Argument:  char **fetch1 +intent(out)
    // Statement: f_out_char**_cdesc_pointer
    void TES_fetchCharPtr_bufferify(void)
    {
        // splicer begin function.fetchCharPtr_bufferify
        fetchCharPtr();
        // splicer end function.fetchCharPtr_bufferify
    }

While it is possible to define every field of the statement group, it
is usually better to build up the group by using mixin groups.  The
mixin groups are also in ``fc-statements.json``.  These groups
encapsulate parts of the wrapper that can be reused by many other
groups.
                
.. literalinclude:: ../shroud/fc-statements.json
   :language: json
   :start-after: "sphinx-start-after": "f_out_char**_cdesc_pointer"
   :end-before: "sphinx-end-before": "f_out_char**_cdesc_pointer"
   :dedent: 8

The command line option :option:`--write-statements` will create a
file with the final form of each statement group.  The final statement
group becomes:

.. literalinclude:: ../regression/reference/none/statements
   :language: yaml
   :start-after: sphinx-start-after: f_out_char**_cdesc_pointer
   :end-before: sphinx-end-before: f_out_char**_cdesc_pointer
    
The final Fortran wrapper become:

.. literalinclude:: ../regression/reference/char-cxx/wrapfchar.f
   :language: fortran
   :start-after: start fetch_char_ptr_library
   :end-before: end fetch_char_ptr_library
   :dedent: 4

With an interface:

.. literalinclude:: ../regression/reference/char-cxx/wrapfchar.f
   :language: fortran
   :start-after: start c_fetch_char_ptr_library
   :end-before: end c_fetch_char_ptr_library
   :dedent: 4

And the C wrapper:

.. literalinclude:: ../regression/reference/char-cxx/wrapchar.cpp
   :language: c++
   :start-after: start CHA_fetchCharPtrLibrary_bufferify
   :end-before: end CHA_fetchCharPtrLibrary_bufferify

A Fortran helper function is also added to properly define the `LEN` of the
`CHARACTER` argument.

.. literalinclude:: ../regression/reference/char-cxx/wrapfchar.f
   :language: fortran
   :start-after: start helper pointer_string
   :end-before: end helper pointer_string
