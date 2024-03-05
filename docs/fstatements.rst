.. Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and
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


A Fortran wrapper is created out of several segments.

.. code-block:: text

      {F_subprogram} {F_name_impl}(f_arg_name)[result (f_result)]
        f_module
        f_arg_decl
        ! splicer begin
        f_declare
        f_pre_call
        f_call {}( {f_arg_call} )
        f_post_call
        ! splicer end
      end {F_subprogram} {F_name_impl}


The ``bind(C)`` interface is defined by the cstatements since it must
match the C wrapper that is being called.  The C wrapper may have a
different API than the Fortran wrapper since the Fortran may pass down
additional arguments.

..        name="f_default",
..        c_helper="",

Format fields
-------------

* F_subprogram
* F_name_impl
* F_arguments
* F_result_clause


statements
----------

f_helper
^^^^^^^^

A list of Fortran helper function names to add to generated
Fortran code.
The format dictionary will be applied to the list for additional
flexibility.

.. code-block:: yaml

    f_helper:
    - array_context

Each helper will add an entry into the format dictionary with
the name of the function or type created by the helper.
The format value is the helper name prefixed by *f_helper_*.
For example,format field *f_helper_array_context* may be ``VEC_SHROUD_array``.

There is no current way to add user defined helper functions.

.. These functions are defined in whelper.py.

f_module
^^^^^^^^

``USE`` statements to add to Fortran wrapper.
A dictionary of list of ``ONLY`` names:

.. code-block:: yaml

        f_module:
          iso_c_binding:
          - C_SIZE_T
   
f_need_wrapper
^^^^^^^^^^^^^^

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

f_arg_name
^^^^^^^^^^

List of name of arguments for Fortran subprogram.
Will be formatted before being used to expand ``{f_var}``.

Any function result arguments will be added at the end.
Only added if *f_arg_decl* is also defined.

f_arg_decl
^^^^^^^^^^

List of argument or result declarations.
Usually constructed from YAML *decl* but sometimes needs to be explicit
to add Fortran attributes such as ``TARGET`` or ``POINTER``.
Also used when a function result is converted into an argument.
Added before splicer since it is part of the API and must not be changed
by the splicer.
Additional declarations can be added within the splicer via *f_declare*.

.. code-block:: text

        f_arg_name=["{f_var}"],
        f_arg_decl=[
            "character, value, intent(IN) :: {f_var}",
        ],

.. result declaration is added before arguments
   but default declaration are after declarations.

It is also used to declare a result defined with *f_result* when
converting a subroutine into a function.

f_arg_call
^^^^^^^^^^

List of arguments to pass to C wrapper.
By default the arguments of the Fortran wrapper are passed to the C
wrapper.  The list of arguments can be set to pass additional
arguments or expressions.  The format field *f_var* the name of the
argument.

When used with a **f_function** statement, the argument will be added
to the end of the call list.

.. code-block:: text

        f_arg_call=[
             "C_LOC({f_var})"
        ],

.. code-block:: text

        f_arg_call=[
            "{f_var}",
            "len({f_var}, kind=C_INT)",
        ],

To specify no arguments, the list must be blank.
Unless the function result has been changed into a C wrapper
argument, it will pass no arguments.

.. code-block:: text

        f_arg_call=[ ],

The value of *None* will pass the Fortran argument
to the C wrapper.

f_declare
^^^^^^^^^

A list of declarations needed by *f_pre_call* or *f_post_call*.
Usually a *c_local_var* is sufficient.
No executable statements should be used since all declarations must be
grouped together.
Implies *f_need_wrapper*.
Added within the splicer to make it easier to replace in the YAML file.

f_module
^^^^^^^^

Fortran modules used in the Fortran wrapper:

.. code-block:: yaml

        f_module:
          iso_c_binding:
          - C_PTR

Fields will be expanded using the format dictionary before being used.

f_pre_call
^^^^^^^^^^

Statement to execute before call, often to coerce types when *f_cast*
cannot be used.
Implies *f_need_wrapper*.
   
f_call
^^^^^^

Code used to call the function.
Defaults to ``{F_result} = {F_C_call}({f_arg_call})``

For example, to assign to an intermediate variable:

.. code-block:: text

        f_declare=[
            "type(C_PTR) :: {c_local_ptr}",
        ],
        f_call=[
            "{c_local_ptr} = {F_C_call}({f_arg_call})",
        ],
        f_local=["ptr"],

.. used with intent function, subroutine, (getter/setter)
   
f_post_call
^^^^^^^^^^^

Statement to execute after call.
Can be use to cleanup after *f_pre_call* or to coerce the return value.
Implies *f_need_wrapper*.
   
f_result
^^^^^^^^

Name of result variable.
Added as the ``RESULT`` clause of the subprogram statement.
Can be used to change a subroutine into a function.

In this example, the subroutine is converted into a function
which will return the number of items copied into the result argument.
*f_arg_decl* is used to declare the result variable.

.. example from vectors.yaml

.. code-block:: yaml

    - decl: void vector_iota_out_with_num2(std::vector<int> &arg+intent(out))
      fstatements:
        f:
          f_result: num
          f_module:
            iso_c_binding: ["C_LONG"]
          f_arg_decl:
          -  "integer(C_LONG) :: num"
          f_post_call:
          -  "num = SHT_arg_cdesc%size"

When set to **subroutine** it will treat the Fortran wrapper as a ``subroutine``.
Used when the function result is passed as an argument to the Fortran wrapper
instead of being returned as the Fortran wrapper result. Typically to avoid
memory allocations by copying directly into the callers variable.

.. deref(arg)

f_temps
^^^^^^^

A list of suffixes for temporary variable names.

.. code-block:: yaml

    f_temps=["len"]

Create variable names in the format dictionary using
``{fmt.c_temp}{rootname}_{name}``.
For example, argument *foo* creates *SHT_foo_len*.

The format field is named *f_var_{name}*.

f_local
^^^^^^^

Similar to *f_temps* but uses ``{fmt.C_local}{rootname}_{name}``.
*temps* is intended for arguments and is typically used in a mixin
group.  *f_local* is used by group to generate names for local
variables.  This allows creating names without conflicting with
*f_temps* from a *mixin* group.

The format field is named *f_local_{name}*.

notimplemented
--------------

If True the statement is not implemented.
The generated function will have ``#if 0`` surrounding the
wrapper.

This is a way to avoid generating code which will not compile when
the notimplemented wrapper is not needed. For example, the C wrapper
for a C++ function when only the C bufferify wrapper is needed for
Fortran.  The statements should eventually be completed to wrap the
function properly.
             
How typemaps are found
----------------------

alias
^^^^^

List of other names which will be used for its contents.

.. code-block:: yaml

        name="fc_out_string_**_cdesc_allocatable",
        alias=[
            "f_out_string_**_cdesc_allocatable",
            "c_out_string_**_cdesc_allocatable",
        ],
