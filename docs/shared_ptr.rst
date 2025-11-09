.. Copyright Shroud Project Developers. See LICENSE file for details.

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
      - type: std::shared_ptr
      - type: std::weak_ptr
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

The *smart_pointer* entry may have a *format* entry to control
names in the generated code.
The name of the shared object is controlled by the format field
*C_name_shared_api* which has a default value from option
*C_name_shared_api_template*.

The ``final`` function on object_shared will call the ``reset``
function to decrement the reference count.

Shroud will add the function ``use_count`` to operate on the shared_ptr.
It can be renamed with the format field *F_name_shared_use_count*.  If
*F_name_shared_use_count* is blank the function will not be added.

.. code-block:: fortran

    type(object_shared) shared
    type(object_weak) weak

    shared = object_shared()
    weak = shared
    print *, shared.use_count()

.. Adding a new smart pointer

   Create a Typemap
      sgroup="smart_ptr"
      smart_pointer="name", used in generated names.
   Add to symtab

   Generated Typemaps for shared_ptr<T>, use sgroup="smartptr"
