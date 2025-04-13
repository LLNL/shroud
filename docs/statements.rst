.. Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
   other Shroud Project Developers.
   See the top-level COPYRIGHT file for details.

   SPDX-License-Identifier: (BSD-3-Clause)

.. _StatementsAnchor:

Statements
==========

The statement data structure is used to define code snippets that will be
used to create the wrapper.
Combinations of language, type and attributes are used to select
a statement name based on a function's argument declaration in the YAML input.
Shroud provides defaults for most standard wrappings.
Statements provides a way for users to have more control over wrapping.
Each aspect of the wrapping, argument names, argument declarations, and passing
to the next layer, must be supplied by the statements in a consistent manner.

.. While information may be redundent across layers, it provides complete control.
   For example, defining both f_result_var and i_result_var in vectors.yaml.

Details for :ref:`C <top_C_Statements>` and :ref:`Fortran <top_Fortran_Statements>`
are provided in other sections.

.. Python format fields....

The command line option ``--write-statements`` can be used to create a
file which will contain all of the statements that Shroud knows.

The option *debug* in the YAML file will add additional comments into
the wrapper to identify which statement names were used to wrap an
argument.

.. code-block:: yaml

    options:
      debug: True


Shared fields
-------------

All statement groups, independent of the wrapper language, share
common fields. These fields are used to identify the group and
to build up complete groups from shared features.

name
^^^^

The name is a underscore delimited list of parts which are used to
find the group.
The first part is the language being wrapped and the second part
is the intent of the group.

Names which start with a ``#`` are ignored.
This provides a way to add comments into the JSON file.
(which does not support comments)

.. code-block:: json

    {
        "name":"##### enum #################################################"
    },
    {
        "name":"f_in_native"
    },
                
alias
^^^^^

The most common way of reusing a group is to create additional names
for the group via an alias.
An alias field can be used with or without the name field.

If there are C and Fortran group, make the first alias a Fortran name.
C is a subset of Fortran and the first alias determines the defaults.

.. code-block:: yaml

        alias:
        - f_out_string**_cdesc_allocatable
        - c_out_string**_cdesc_allocatable

Names which start with a ``#`` are ignored.
This provides a way to add comments into the JSON file.
(which does not support comments)

mixin
^^^^^

When the group has the intent *mixin* in its name it is can be used to
add subsets of fields to other groups.  The fields will be "mixed in"
to another group to avoid repeating fields.
After the language and intent parts of the name, any words can be
appended to the name to help identify the group.

When multiple mixins are used, 
scalar fields are assigned, replacing any existing value in the group
being defined.  A field which is a list will be appended to the group
being defined making it possible to build up a field from several
mixins.  Dictionaries are recursively appended.

.. dictionary example (f_module).

After the mixins are used to initialize a group, if a field name is
reused it will replace the value from the mixin group. This allows a group to use
mixins yet still be customized as needed.

A mixin group must not contain *alias*, *append* or *base*

.. code-block:: yaml

    - name: f_mixin_one
      f_dummy_decl:
      - integer arg1
      f_arg_call:
      -  arg1
      f_need_wrapper: True
    - name: f_mixin_two
      f_dummy_decl:
      - integer arg2
      f_arg_call:
      -  arg2
      f_need_wrapper: False
    - name: f_in_type
      mixin:
      -  f_mixin_one
      -  f_mixin_two
      f_arg_call:
      -  arg1 + arg2

The final group will be:

.. code-block:: yaml

    - name: f_in_type
      f_dummy_decl:
      - integer arg1
      - integer arg2
      f_arg_call:
      -  arg1 + arg2
      f_need_wrapper: False

While this example uses twice as many lines to create the *f_in_type*
group, the real benefit is when each mixin group contains several
declarations and is mixed into many other groups.

Names which start with a ``#`` are ignored.
This provides a way to add comments into the JSON file.
(which does not support comments)

comments
^^^^^^^^

Comments list the steps used by the wrapper.
The final group's comment will be a collection of all of the mixin's comments.
Each mixin can contribute a step in the wrapping process.
Comments are appended as part of the mixin process.
Note that if the final non-mixin group also defines comments, it will replace
the comments created from the mixins. *notes* is intended for text which
provides additional information specific to a group.

notes
^^^^^

Notes are used provide usage notes for a group.
Notes are not mixed into groups.

.. base - must be single name.
          Applied after all of the others mixins as parent of Scope.
          Cannot also have a *mixin* field.
          Useful to define a group that varies slightly
          such as pointer vs reference argument.

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

usage
^^^^^

Documents a typical declarion which will use this group.

.. code-block:: yaml

        name: "f_out_char**_cdesc_pointer"
        usage: [
            "char **arg +intent(out)"
        ]
                
fmtdict
^^^^^^^

A dictionary to replace default values

.. code-block:: yaml

        name: f_mixin_getter_argname
        fmtdict:
            f_var: "val"
            i_var: "val"
            c_var: "val"

.. code-block:: yaml

        name: c_mixin_function-assign-to-new
        fmtdict:
            cxx_addr: ""
            cxx_member: "->"
                
Examples
--------

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

* intent - ``in``, ``out``, ``inout``, ``function``, ``subroutine``, ``ctor``, ``dtor``, ``getter``, ``setter``

* Abstract declaration. For example, ``native``, ``native*`` or ``native**``.
  May include template arguments ``vector<native>``.
  Uses the typemap field **sgroup**.

* api - from attribute
  ``buf``, ``capsule``, ``capptr``, ``cdesc`` and ``cfi``.

* funcarg - from attribute.
  Uses ``funcarg`` and not the value of the attribute.
  
* deref - from attribute
  ``allocatable``, ``pointer``, ``raw``, ``scalar``

* owner
  ``caller``, ``library``

* operator
  ex. ``assignment``

* custom
  ex. ``weakptr``

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

Each statement group is evaluated in the context of a format
dictionary created for the function result or argument.

* c_arglist, f_arglist

  An array of format fields for all arguments to the function.  Entry
  0 is the function, 1 is the first argument, and so on.  This allows
  a statement group to access other variable using a value such as
  ``c_arglist[1].c_local_cxx``.

Several format fields are defined to help use a set of statements with both
pointers and references.

* cxx_member

.. code-block:: yaml

    "c_post_call": [
        "{c_const}char *{c_var} = {cxx_var}{cxx_member}c_str();"
    ]



Some attributes can be used to generate strings based on the declaration and attributes.

.. defined in fcfmt.py

Dimension
^^^^^^^^^

Since the rank of arrays can vary, the attribute can create a varying number of lines.

.. Move these into section talking about cdesc or CFI.

* gen.f_allocate_shape

  Shape to use with ``ALLOCATE`` statement from ``cdesc`` variable.
  Blank if scalar.

* gen.c_f_pointer

  Shape for ``C_F_POINTER`` intrinsic from ``cdesc`` variable.
  Blank for scalars.

* gen.f_cdesc_shape

  Assign variable shape to ``cdesc`` in Fortran using ``SHAPE`` intrinsic.
  This will be passed to C wrapper.
  Blank for scalars.
  
* gen.c_dimension_size

  Compute size of array from *dimension* attribute.
  ``1`` if scalar.
  
* gen.c_array_shape

  Assign array shape to a *cdesc* variable in C.
  Blank if scalar.
  
* gen.c_array_size
  
  Return expression to compute the size of an array.
  `*c_array_shape* must be used first to define ``c_var_cdesc->shape``.
  ``1`` if scalar.

* gen.c_extents_decl

  Define the shape in local variable extents
  in a ``CFI_index_t`` variable.
  Blank if scalar.
  
* gen.c_extents_use
  
  Return variable name of extents of CFI array.
  ``NULL`` if scalar.

* gen.c_lower_use
  
  Return variable name of lower bounds of CFI array
  from helper *lower_bounds_CFI*.
  ``NULL`` if scalar.

Examle with CFI arrays.
  
.. literalinclude:: ../shroud/fc-statements.json
   :language: json
   :start-after: "sphinx-start-after": "c_mixin_cfi_native_allocatable"
   :end-before: "sphinx-end-before": "c_mixin_cfi_native_allocatable"
   :dedent: 8

.. mixin naming conventions
   Must start with c_mixin_ or f_mixin_
   Group together by adding cdesc_  or capsule_ or _cfi.
   This makes the mixin block standout better which groups are working together
   rather than adding the intent or type before cdesc or capsule.
   
   The non-mixin groups add capsule/cdesc later since it is from api(cdesc).

   When using cdesc as an argument, declare the local variable using cxx_var,
   not cxx_local_var. This is the name used in the declaration.
   For example, c_mixin_local-string* which defines std::string.
