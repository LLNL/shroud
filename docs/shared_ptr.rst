.. Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and
   other Shroud Project Developers.
   See the top-level COPYRIGHT file for details.

   SPDX-License-Identifier: (BSD-3-Clause)

.. _Shared_PtrAnchor:


Shared pointers
===============

.. note:: Work in progress

Shroud has support for ``std::shared_ptr``.

A ``std::shared_ptr`` will be created when the constructor has the
**+owner(shared)** attribute.
The option **C_shared_ptr** is used to create a ``FINAL`` subprogram
which will reduce the count in the ``shared_ptr``.

