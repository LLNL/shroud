.. Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and
   other Shroud Project Developers.
   See the top-level COPYRIGHT file for details.

   SPDX-License-Identifier: (BSD-3-Clause)

.. _Shared_PtrAnchor:


Shared pointers
===============

.. note:: Work in progress

Shroud has support for smart pointers.
When the entry **smart_pointer** has values in a class, a subclass will be
created in the Fortran wrapper. This shadow class holds the smart pointer instead
of a pointer to an instance of the class.

.. code-block:: yaml

    declarations:
    - decl: class Object
      smart_pointer:
      - name: std::shared_ptr
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

The name of the shared object is controlled by the format field
*C_name_shared_api* which has a default value from option
*C_name_shared_api_template*.

The ``final`` function on object_shared will call the ``reset``
function to decrement the reference count.

Shroud will add the function ``use_count`` to operate on the shared_ptr.
It can be renamed with the format field *F_name_shared_use_count*.  If
*F_name_shared_use_count* is blank the function will not be added.
