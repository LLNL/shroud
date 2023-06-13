.. Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and
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
        call {}( {F_arg_c_call} )
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

Shroud tries to only create an interface for a C function to
avoid the extra layer of a Fortran wrapper.
However, often the Fortran wrapper needs to do some work that
the C wrapper cannot.
This field can be set to True to ensure the Fortran wrapper
is created.
This is used when an assignment is needed to do a type coercion;
for example, with logical types.

A wrapper will always be created if the **F_force_wrapper**
option is set.

.. XXX tends to call bufferify version

arg_name
^^^^^^^^

List of name of arguments for Fortran subprogram.
Will be formatted before being used to expand ``{f_var}``.

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
By default the arguments of the Fortran wrapper are passed to the C
wrapper.  The list of arguments can be set to pass additional
arguments or expressions.  The format field *f_var* the name of the
argument.

When used with a **f_function** statement, the argument will be added
to the end of the call list.

.. code-block:: text

        arg_c_call=[
             "C_LOC({f_var})"
        ],
        arg_c_call=[
            "{f_var}",
            "len({f_var}, kind=C_INT)",
        ],

c_local_var
^^^^^^^^^^^

If *true* an intermediate variable is created then passed to the C
wrapper instead of passing *f_var* directly.  The intermediate
variable can be used when the Fortran argument must be processed
before passing to C.

For example, the statements for **f_in_bool** convert the type from
``LOGICAL`` to ``logical(C_BOOL)``. There is no intrinsic function to
convert logical variables so an assignment statement is required to
cause the compiler to convert the value.

.. code-block:: yaml

    dict(
        name="f_in_bool",
        c_local_var=True,
        pre_call=["{c_var} = {f_var}  ! coerce to C_BOOL"],
    ),

.. XXX - maybe use *temps* and *f_c_arg_names* instead as a more general solution.

declare
^^^^^^^

A list of declarations needed by *pre_call* or *post_call*.
Usually a *c_local_var* is sufficient.
No executable statements should be used since all declarations must be
grouped together.
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
            "type(C_PTR) :: {c_local_ptr}",
        ],
        call=[
            "{c_local_ptr} = {F_C_call}({F_arg_c_call})",
        ],
        local=["ptr"],
                
   
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

temps
^^^^^

A list of suffixes for temporary variable names.

.. code-block:: yaml

    temps=["len"]

 Create variable names in the format dictionary using
 ``{fmt.c_temp}{rootname}_{name}``.
 For example, argument *foo* creates *SHT_foo_len*.

local
^^^^^

 Similar to *temps* but uses ``{fmt.C_local}{rootname}_{name}``.
 *temps* is intended for arguments and is typically used in a mixin
 group.  *local* is used by group to generate names for local
 variables.  This allows creating names without conflicting with
 *temps* from a *mixin* group.


             
How typemaps are found
----------------------

alias
^^^^^

Names another node which will be used for its contents.
