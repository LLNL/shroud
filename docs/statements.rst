.. Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and
   other Shroud Project Developers.
   See the top-level COPYRIGHT file for details.

   SPDX-License-Identifier: (BSD-3-Clause)


Statements
==========

Shroud can be thought of as a fancy macro processor.
The statement data structure is used to define code that should be
used to create the wrapper.
Combinations of language, type and attributes are used to select
a statement entry.


.. mixin - must be a single name, no alternative allowed such as allocatable/pointer


Passing function result as an argument
--------------------------------------

This section explains how statements are used to generate code for
functions which return a struct.

Compiler ABI do not agree on how some function results should be
returned.  To ensure portablity, some function results must be passed
as an additional argument.  This is typically more complicated types
such as struct or complex.

.. literalinclude:: ../shroud/statements.py
   :language: python
   :start-after: start function_struct_scalar
   :end-before: end function_struct_scalar

