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


.. base - must be single name.
          Applied after all of the others mixins as parent of Scope.
          Cannot also have a *mixin* field.
          Useful to define a group that varies slightly
          such as pointer vs reference argument.

.. mixin - list of single names, no alternative allowed such as allocatable/pointer
           must not contain 'alias' or 'base'
           List fields from the mixin group will be appended to the group
           being defined.
           Non-lists are assigned.

.. append - applied after mixins as a sort of one-off mixin to append to fields.
      f_post_call is defined by the mixins but need to add one more line.

        name="f_out_string_**_cdesc_allocatable",
        mixin=[
            "f_mixin_out_array_cdesc",
            "f_mixin_out_array_cdesc_allocatable",
        ],
        append=dict(
            f_post_call=[
                "call {fhelper_array_string_allocatable}({f_var_alloc}, {f_var_cdesc})",
            ],
        ),

        or maybe with {copy_allocate} in the mixin.

        fmtdict:
           copy_allocate: "call {fhelper_array_string_allocatable}({f_var_alloc}, {f_var_cdesc})"
   

.. fmtdict - A dictionary to replace default values

        name: f_function_char_*_cfi_arg
        base: f_function_char_*_cfi_copy
        fmtdict:
            f_var: "{F_string_result_as_arg}"
            c_var: "{F_string_result_as_arg}"
   

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

