.. Copyright (c) 2017-2021, Lawrence Livermore National Security, LLC and
   other Shroud Project Developers.
   See the top-level COPYRIGHT file for details.

   SPDX-License-Identifier: (BSD-3-Clause)


Fortran Statements
==================

.. note:: Work in progress.

Typemaps are used to add code to the generated wrappers
to replace the default code.

The statements work together to pass variables and metadata between
Fortran and C.


fc_statements
-------------

A Fortran wrapper is created out of several segments.

.. code-block:: text

      {F_subprogram} {F_name_impl}({F_arguments}){F_result_clause}
        arg_f_use
        arg_f_decl
        ! splicer begin
        declare
        pre_call
        call
        post_call
        ! splicer end
      end {F_subprogram} {F_name_impl}


The ``bind(C)`` interface is defined by the cstatements since it must
match the C wrapper that is being called.  The C wrapper may have a
different API than the Fortran wrapper since the Fortran may pass down
additional arguments.

..        name="f_default",
..        c_helper="",
..        c_local_var=None,

f_helper
^^^^^^^^

Blank delimited list of Fortran helper function names to add to generated
Fortran code.
These functions are defined in whelper.py.
There is no current way to add user defined helper functions.

f_module
^^^^^^^^

``USE`` statements to add to Fortran wrapper.
A dictionary of list of ``ONLY`` names:

.. code-block:: yaml

        f_module:
          iso_c_binding:
          - C_SIZE_T
   
need_wrapper
^^^^^^^^^^^^

If true, the Fortran wrapper will always be created.
This is used when an assignment is needed to do a type coercion;
for example, with logical types.

.. XXX tends to call bufferify version

arg_name
^^^^^^^^

List of name of arguments for Fortran subprogram.
Will be formated before use to allow ``{f_var}``.

Any function result arguments will be added at the end.
Only added if *arg_decl* is also defined.

arg_decl
^^^^^^^^

List of argument or result declarations.
Usually constructed from YAML *decl* but sometimes needs to be explicit
to add Fortran attributes such as ``TARGET`` or ``POINTER``.
Added before splicer since it is part of the API and must not change.
Additional declarations can be added within the splicer via *declare*.

.. code-block:: text

        arg_decl=[
            "character, value, intent(IN) :: {f_var}",
        ],

.. result declaration is added before arguments
   but default declaration are after declarations.

arg_c_call
^^^^^^^^^^

List of arguments to pass to C wrapper.
By default the arguments of the Fortran wrapper are passed to the C wrapper.
The list can be modified to pass additional arguments or expressions.

.. code-block:: text

        arg_c_call=[
             "C_LOC({f_var})"
        ],
        arg_c_call=[
            "{f_var}",
            "len({f_var}, kind=C_INT)",
        ],

declare
^^^^^^^

A list of declarations needed by *pre_call* or *post_call*.
Usually a *c_local_var* is sufficient.
Implies *need_wrapper*.
Added within the splicer to make it easier to replace in the YAML file.

f_import
^^^^^^^^

List of names to import into the Fortran wrapper.
The names will be expanded before being used.

In this example, Shroud creates *F_array_type* derived type in the
module and it is used in the interface.

.. code-block:: yaml

        f_import=["{F_array_type}"],
                
f_module
^^^^^^^^

Fortran modules used in the Fortran wrapper:

.. code-block:: yaml

        f_module=dict(iso_c_binding=["C_PTR"]),

f_module_line
^^^^^^^^^^^^^

Fortran modules used in the Fortran wrapper as a single line
which allows format strings to be used.

.. code-block:: yaml

        f_module_line="iso_c_binding:{f_kind}",

The format is::

     module ":" symbol [ "," symbol ]* [ ";" module ":" symbol [ "," symbol ]* ]

pre_call
^^^^^^^^

Statement to execute before call, often to coerce types when *f_cast*
cannot be used.
Implies *need_wrapper*.
   
call
^^^^

Code used to call the function.
Defaults to ``{F_result} = {F_C_call}({F_arg_c_call})``

For example, to assign to an intermediate variable:

.. code-block:: text

        declare=[
            "type(C_PTR) :: {F_pointer}",
        ],
        call=[
            "{F_pointer} = {F_C_call}({F_arg_c_call})",
        ],
                
   
post_call
^^^^^^^^^

Statement to execute after call.
Can be use to cleanup after *pre_call* or to coerce the return value.
Implies *need_wrapper*.
   
result
^^^^^^

Name of result variable.
Added as the ``RESULT`` clause of the subprogram statement.
Can be used to change a subroutine into a function.

In this example, the subroutine is converted into a function
which will return the number of items copied into the result argument.

.. code-block:: yaml

    - decl: void vector_iota_out_with_num2(std::vector<int> &arg+intent(out))
      fstatements:
        f:
          result: num
          f_module:
            iso_c_binding: ["C_LONG"]
          declare:
          -  "integer(C_LONG) :: num"
          post_call:
          -  "num = Darg%size"


How typemaps are found
----------------------

alias
^^^^^

Names another node which will be used for its contents.
