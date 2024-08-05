.. Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and
   other Shroud Project Developers.
   See the top-level COPYRIGHT file for details.

   SPDX-License-Identifier: (BSD-3-Clause)

.. _StatementsAnchor:

Statements
==========

Shroud can be thought of as a fancy macro processor.
The statement data structure is used to define code snippets that should be
used to create the wrapper.
Combinations of language, type and attributes are used to select
a statement name based on the YAML input.

Details for :ref:`C <top_C_Statements>` and :ref:`Fortran <top_Fortran_Statements>`
are provided in other sections.


Python format fields....


Statement names or alias which start with a ``#`` are ignored.
This provides a way to add comments into the JSON file.
(which does not support comments)

.. code-block:: json

    {
        "name":"##### enum #################################################"
    },
                
.. name

.. comments
   Comments are appended as part of the mixin process.
   The final group's comment will be a collection of all of the mixin's comments.
   Each mixin can contribute a step in the wrapping process.

.. notes
   Notes are used provide usage notes for a group.
   Notes are not mixed into groups.

.. base - must be single name.
          Applied after all of the others mixins as parent of Scope.
          Cannot also have a *mixin* field.
          Useful to define a group that varies slightly
          such as pointer vs reference argument.

.. mixin - list of names
           List fields from the mixin group will be appended to the group
           being defined.
           Non-lists are assigned.
           Dictionaries are recursively appended (f_module).

           A mixin group is created when the intent in the name is 'mixin'.
           Must not contain 'alias', 'append' or 'base'

.. append - applied after mixins as a sort of one-off mixin to append to fields.
      f_post_call is defined by the mixins but need to add one more line.
      Cannot be used in a mixin.

        name="f_out_string_**_cdesc_allocatable",
        mixin=[
            "f_mixin_out_array_cdesc",
            "f_mixin_out_array_cdesc_allocatable",
        ],
        append=dict(
            f_post_call=[
                "call {f_helper_array_string_allocatable}({f_var_alloc}, {f_var_cdesc})",
            ],
        ),
        f_post_call [ ]      # will replace the value instead of appending.

        or maybe with {copy_allocate} in the mixin.

        fmtdict:
           copy_allocate: "call {f_helper_array_string_allocatable}({f_var_alloc}, {f_var_cdesc})"

.. alias
     An alias field can be used with or without the name field.

     If there are C and Fortran group, make the first alias a Fortran name.
     C is a subset of Fortran and the first alias determines the defaults.

.. fmtdict - A dictionary to replace default values

        name: f_function_char_*_cfi_arg
        base: f_function_char_*_cfi_copy
        fmtdict:
            f_var: "{F_string_result_as_arg}"
            c_var: "{F_string_result_as_arg}"


.. code-block:: yaml

    name: f_mixin_one
    f_pre_call:
    - "! comment f_mixin_one"

    name: f_mixin_two
    mixin:
    - f_mixin_one
    f_pre_call:
    - "! comment f_mixin_two"    # appends

    name: f_function_one
    mixin:
    - f_mixin_one
    f_pre_call:
    - "! comment two"            # replaces


Lookup statements
-----------------

The statements for an argument are looked up by converting the type
and attributes into an underscore delimited string.


* language - ``c``

* intent - ``in``, ``out``, ``inout``, ``function``, ``ctor``, ``dtor``, ``getter``, ``setter``

* Abstract declaration. For example, ``native``, ``native*`` or ``native**``.
  May include template arguments ``vector<native>``.
  Uses the typemap field **sgroup**.

* api - from attribute
  ``buf``, ``capsule``, ``capptr``, ``cdesc`` and ``cfi``.

* deref - from attribute
  ``allocatable``, ``pointer``, ``raw``, ``scalar``


Common Statement Fields
-----------------------

name
^^^^

A name can contain variants separated by ``/``.

.. code-block:: yaml

    - name: c_in/out/inout_native_*_cfi

This is equivelent to having three groups:
    
.. code-block:: yaml

    - name: c_in_native_*_cfi
    - name: c_out_native_*_cfi
    - name: c_inout_native_*_cfi

alias
^^^^^

List of other names which will be used for its contents.

.. code-block:: yaml

        name="fc_out_string_**_cdesc_allocatable",
        alias=[
            "f_out_string_**_cdesc_allocatable",
            "c_out_string_**_cdesc_allocatable",
        ],

comments
^^^^^^^^

notes
^^^^^

Passing function result as an argument
--------------------------------------

This section explains how statements are used to generate code for
functions which return a struct.

Compiler ABI do not agree on how some function results should be
returned.  To ensure portablity, some function results must be passed
as an additional argument.  This is typically more complicated types
such as struct or complex.

.. literalinclude:: ../shroud/fc-statements.json
   :language: json
   :start-after: start function_struct_scalar
   :end-before: end function_struct_scalar


Classes and Structs
-------------------

The default behavior for classes and structs is to pass them through
from Fortran to C++ without looking inside them. The statements are
selected based on the typemap's sgroup field which is either *shadow*
or *struct*.

In some cases, it can be beneficial to look inside a compound type.
By setting the option *typemap_sgroup*, a statement group can be used
which is specific for the type. A prime example of this is
``std::vector``. This maps naturally to a Fortran array. The C wrapper
accepts a pointer to the array along with a length argument. The C
wrapper is then responsible for creating the ``std::vector``. If the
``std::vector`` was created in the Fortran code it could be passed
opaquely; however, it would be very inconvenient to access elements in
Fortran requiring the use of the ``vector.at`` method instead of
Fortran array subscripting.

See the sgroup.yaml test.

.. f_in_twostruct<native,native> vs f_in_struct<native,native>.


Format fields
-------------

Several format fields are defined to help use a set of statements with both
pointers and references.

* cxx_member

.. code-block:: yaml

    "c_post_call": [
        "{c_abstract_decl} {c_var} = {cxx_var}{cxx_member}c_str();"
    ]
