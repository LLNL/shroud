.. Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
   other Shroud Project Developers.
   See the top-level COPYRIGHT file for details.

   SPDX-License-Identifier: (BSD-3-Clause)

Namespaces
==========

Namespaces in C++ are used to ensure the symbols in a library will not
conflict with any symbols in another library.  Fortran and Python both
use a module to accomplish the same thing.

The global variable *namespace* is a blank delimited list of
namespaces used as the initial namespace.  This namespace will be used
when accessing symbols in the library, but it will not be used when
generating names for wrapper functions.

For example, the library ``wrapped`` is associated with the namespace
``outer``.  There are three functions all with the same name,
``worker``.  In C++ these functions are accessed by using a fully
qualified name: ``outer::worker``, ``outer::innter1::worker`` and
``outer::inner2::worker``.

.. code-block:: c++

    namespace outer {
      namespace inner1 {
        void worker();
      } // namespace inner1

      namespace inner2 {
        void worker();
      }  // namespace inner2

      void worker();
    } // namespace outer


The YAML file would look like:

.. code-block:: yaml

    library: wrapped
    namespace: outer
    format:
      C_prefix: WWW_

    declarations:
    - decl: namespace inner1
      declarations:
      - decl: void worker();
    - decl: namespace inner2
      declarations:
      - decl: void worker();
    - decl: void worker();

For each namespace, Shroud will generate a C++ header file, a C++
implementation file, a Fortran file and a Python file.
The nested namespaces are added to the format field *C_name_scope*.

For the C wrapper, all symbols are globally visible and must be
unique. The format fields *C_prefix* and *C_name_scope* are used to
generate the names. This will essentially "flatten" the namespaces into
legal C identifiers.

.. code-block:: c

    void WWW_worker();
    void WWW_inner1_worker();
    void WWW_inner2_worker();

In Fortran each namespace creates a module.  Each module will have
a function named *worker*. This makes the user responsible for distinguising
which implementation of *worker* is to be called.

.. code-block:: fortran

    subroutine work1
      ! Use a single module, unambiguous
      use wrapped_mod
      call worker
    end subroutine work1

    subroutine work2
      ! Rename symbol from namespace inner1
      use wrapped_mod
      use wrapped_inner1_mod, inner_worker => worker
      call worker
      call inner_worker
    end subroutine work2

.. options flatten_namespace

Each namespace creates a Python module.

.. code-block:: python

    import wrapped
    wrapped.worker()
    wrapped.inner1.worker()

.. These fields are also in reference.rst

Several fields in the format dictionary are updated for each namespace:
*namespace_scope*, *C_name_scope*, *F_name_scope*.


std namespace
-------------

Shroud has builtin support for ``std::string`` and ``std::vector``.
