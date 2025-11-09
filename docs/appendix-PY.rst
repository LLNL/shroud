.. Copyright Shroud Project Developers. See LICENSE file for details.

   SPDX-License-Identifier: (BSD-3-Clause)

.. ############################################################

.. _pyexample_Numpy Struct Descriptor:

Numpy Struct Descriptor
^^^^^^^^^^^^^^^^^^^^^^^

:file:`struct.yaml`:

.. code-block:: yaml

    - decl: struct Cstruct1 {
              int ifield;
              double dfield;
            };

.. literalinclude:: ../regression/reference/struct-numpy-c/pystructmodule.c
   :language: c
   :start-after: start PY_Cstruct1_create_array_descr
   :end-before: end PY_Cstruct1_create_array_descr

