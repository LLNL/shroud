.. Copyright (c) 2017-2021, Lawrence Livermore National Security, LLC and
   other Shroud Project Developers.
   See the top-level COPYRIGHT file for details.

   SPDX-License-Identifier: (BSD-3-Clause)

Templates
---------

Shroud will wrap templated classes and functions for explicit instantiations.
The template is given as part of the ``decl`` and the instantations are listed in the
``cxx_template`` section:

.. code-block:: yaml

  - decl: |
        template<typename ArgType>
        void Function7(ArgType arg)
    cxx_template:
    - instantiation: <int>
    - instantiation: <double>

``options`` and ``format`` may be provide to control the generated code:

.. code-block:: yaml

  - decl: template<typename T> class vector
    cxx_header: <vector>
    cxx_template:
    - instantiation: <int>
      format:
        C_impl_filename: wrapvectorforint.cpp
      options:
        optblah: two
    - instantiation: <double>

.. from templates.yaml

For a class template, the *class_name* is modified to included the
instantion type.  If only a single template parameter is provided,
then the template argument is used.  For the above example,
*C_impl_filename* will default to ``wrapvector_int.cpp`` but has been
explicitly changed to ``wrapvectorforint.cpp``.

