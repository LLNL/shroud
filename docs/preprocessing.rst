.. Copyright Shroud Project Developers. See LICENSE file for details.

   SPDX-License-Identifier: (BSD-3-Clause)

.. _PreprocessingAnchor:

Preprocessing
=============

Shroud can insert preprocessing directives into the generated source
to implement conditional compilation.  This allows the wrapper to be
generated once, and the source distributed, then compiled to include
only the features available.  For example, with or without MPI.

Most Fortran compiles have an option to use a variant of the C
preprocessor before compiling the source.
To aid portability, only ``#if`` and ``#endif`` are added to the
Fortran source.

It may be useful to set the option
``F_filename_suffix`` to ``F``. This will cause most compilers to
process the Fortran souce with ``cpp`` before compilation.
The alternative is to pass a flag explicit to the compiler
such as ``-fpp`` for intel or ``-cpp`` for gfortran.


The ``fortran_header`` field can be added to the YAML file to
insert ``#include`` directives at the top of the Fortran source files.



The main feature in Shroud is the ``cpp_if`` field for functions and
types.  But the user is always free to add explicit preprocessing
directives via a splicer block.

Functions
---------

.. code-block:: yaml

    - decl: void testmpi(MPI_Comm comm)
      format:
        function_suffix: _mpi
      cpp_if: ifdef HAVE_MPI
    - decl: void testmpi()
      format:
        function_suffix: _serial
      cpp_if: ifndef HAVE_MPI

The function wrappers will be created within ``#ifdef``/``#endif``
directives.  This includes the C wrapper, the Fortran interface and
the Fortran wrapper.  The generated Fortran interface will be:

.. code-block:: fortran

        interface testmpi
    #ifdef HAVE_MPI
            module procedure testmpi_mpi
    #endif
    #ifndef HAVE_MPI
            module procedure testmpi_serial
    #endif
        end interface testmpi

Class generic type-bound function will also insert conditional
compilation directives:

.. code-block:: yaml

    - decl: class ExClass3
      cpp_if: ifdef USE_CLASS3
      declarations:
      - decl: void exfunc()
        cpp_if: ifdef USE_CLASS3_A
      - decl: void exfunc(int flag)
        cpp_if: ifndef USE_CLASS3_A

The generated type will be:

.. code-block:: fortran

        type exclass3
            type(SHROUD_capsule_data), private :: cxxmem
        contains
            procedure :: exfunc_0 => exclass3_exfunc_0
            procedure :: exfunc_1 => exclass3_exfunc_1
    #ifdef USE_CLASS3_A
            generic :: exfunc => exfunc_0
    #endif
    #ifndef USE_CLASS3_A
            generic :: exfunc => exfunc_1
    #endif
        end type exclass3

A ``cpp_if`` field in a class will add a conditional directive around
the entire class.

Finally, ``cpp_if`` can be used with types. This would be required in
the first example since ``mpi.h`` should only be included when
``USE_MPI`` is defined:

.. code-block:: yaml

    typemaps:
    - type: MPI_Comm
      fields:
        cpp_if: ifdef USE_MPI


The ``typemaps`` field can only appear at the outermost layer
and is used to augment existing typemaps.



