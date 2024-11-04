.. Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and
   other Shroud Project Developers.
   See the top-level COPYRIGHT file for details.

   SPDX-License-Identifier: (BSD-3-Clause)

.. _Shared_PtrAnchor:


Shared pointers
===============

.. note:: Work in progress

Shroud has support for ``std::shared_ptr``.
When the option **C_shared_ptr** is set to *true* for a class, a subclass will be
created in the Fortran wrapper. This shadow class holds the smart pointer instead
of a pointer to an instance of the class.

.. code-block:: yaml

    declarations:
    - decl: class Object
      options:
        C_shared_ptr: true
      declarations:
      - decl: Object()
      - decl: ~Object()

The generated Fortran wrappers will be

.. code-block:: fortran

    type object
        type(SHA_SHROUD_capsule_data) :: cxxmem
    contains
        procedure :: dtor => object_dtor
    end type object

    type, extends(object) :: object_shared
    contains
        procedure :: dtor => object_shared_dtor
        final :: object_shared_final
    end type object_shared

    interface object
        module procedure object_ctor
    end interface object

    interface object_shared
        module procedure object_shared_ctor
    end interface object_shared

To create an object in C++ with ``new object``, call the ``object``
function.  To create a shared object with
``std::make_shared(object)``, call the ``object_shared`` function.

The ``final`` function on object_shared will call the ``reset``
function to decrement the reference count.
