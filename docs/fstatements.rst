.. Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
   other Shroud Project Developers.
   See the top-level COPYRIGHT file for details.

   SPDX-License-Identifier: (BSD-3-Clause)

.. _top_Fortran_Statements:

Fortran Statements
==================

.. note:: Work in progress.

Statements are used to add code to the generated wrappers.

The statements work together to pass variables and metadata between
Fortran and C.


A Fortran wrapper is created out of several segments.

``{}`` denotes a format field. ``[]`` is a statement field.

.. code-block:: text

      {F_subprogram} {F_name_impl}({F_arguments}){F_result_clause}
        [f_module]
        [f_dummy_decl]
        ! splicer begin
        [f_local_decl]
        [f_pre_call]
        [f_call]
        [f_post_call]
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
* f_call_function
* F_result_clause

  For functions, ``result({F_result})``.
  Evalued after *stmt.fmtdict* is applied.
  
* F_arg_c_call

Statements
----------

name
^^^^

Must start with a ``f``.

f_dummy_arg
^^^^^^^^^^^

List of dummy argument names for the Fortran subprogram.
It will be formatted before being used to expand ``{f_var}``.

Any function result arguments will be added at the end.
Only added if *f_dummy_decl* is also defined.

f_dummy_decl
^^^^^^^^^^^^

List of dummy argument declarations or result declarations.
Usually constructed from YAML *decl* but sometimes needs to be explicit
to add Fortran attributes such as ``TARGET`` or ``POINTER``.
Also used when a function result is converted into an argument.
Added before splicer since it is part of the API and must not be changed
by the splicer.
Local variables can be declared with *f_local_decl* for function
results or intermediate values.

.. code-block:: yaml

        f_dummy_arg:
        - "{f_var}"
        f_dummy_decl:
        - character, value, intent(IN) :: {f_var}

.. result declaration is added before arguments
   but default declaration are after declarations.

It is also used to declare a result defined with *f_result_var* when
converting a subroutine into a function.

f_local_decl
^^^^^^^^^^^^

A list of declarations needed by *f_pre_call* or *f_post_call*.
Usually a *c_local_var* is sufficient.
No executable statements should be used since all declarations must be
grouped together.
Implies *f_need_wrapper*.
Added within the splicer to make it easier to replace in the YAML file.

f_pre_call
^^^^^^^^^^

Statement to execute before call, often to coerce types when *f_cast*
cannot be used.
Implies *f_need_wrapper*.
   
f_arg_call
^^^^^^^^^^

List of arguments to pass to C wrapper.
By default the arguments of the Fortran wrapper are passed to the C
wrapper. It will use the *f_to_c* typemap field if defined.
The list of arguments can be set to pass different
arguments or expressions. For example, when passing the character length
for attribute ``+api(buf)``.
The format field *f_var* is the name of the argument.

When used with a **f_function** statement, the argument will be added
to the end of the call list.

.. code-block:: yaml

        f_arg_call:
        -  "C_LOC({f_var})"

.. code-block:: yaml

        f_arg_call:
        - "{f_var}"
        -  "len({f_var}, kind=C_INT)"

To specify no arguments, the list must be blank.
Unless the function result has been changed into a C wrapper
argument, it will pass no arguments.

.. code-block:: yaml

        f_arg_call: [ ]

The value of *None* will pass the Fortran argument
to the C wrapper.

f_call
^^^^^^

Code used to call the function.
Defaults to ``{F_result} = {F_C_call}({f_arg_call})``

For example, to assign to an intermediate variable:

.. code-block:: yaml

        f_local_decl:
        - "type(C_PTR) :: {c_local_ptr}"
        f_call:
        - "{c_local_ptr} = {F_C_call}({f_arg_call})"
        f_local:
        - ptr

.. used with intent function, subroutine, (getter/setter)
   
f_post_call
^^^^^^^^^^^

Statement to execute after call.
Can be use to cleanup after *f_pre_call* or to coerce the return value.
Implies *f_need_wrapper*.
   
f_result_var
^^^^^^^^^^^^

Name of result variable.
Added as the ``RESULT`` clause of the subprogram statement.
Can be used to change a subroutine into a function by setting the
value to ``as-subroutine`` (which is an illegal identifier).

In this example, the subroutine is converted into a function
which will return the number of items copied into the result argument.
*f_dummy_decl* is used to declare the result variable.

.. example from vectors.yaml

.. code-block:: yaml

    - decl: void vector_iota_out_with_num2(std::vector<int> &arg+intent(out))
      fstatements:
        f:
          f_result_var: num
          f_dummy_decl:
          -  "integer(C_LONG) :: num"
          f_post_call:
          -  "num = SHT_arg_cdesc%size"
          f_module:
            iso_c_binding: ["C_LONG"]

When set to **subroutine** it will treat the Fortran wrapper as a ``subroutine``.
Used when the function result is passed as an argument to the Fortran wrapper
instead of being returned as the Fortran wrapper result. Typically to avoid
memory allocations by copying directly into the callers variable.

.. deref(arg)

f_module
^^^^^^^^

``USE`` statements to add to Fortran wrapper.
A dictionary of list of ``ONLY`` names.
The names will be expanded before being uses for format values can be used.

If ``typemap.f_module_name` is None, then the default
``format.f_module_name`` value will be used. No module will be used in
the wrapper in this case.  For example, ``CHARACTER`` which does not
use a kind in the declaration since it uses the Fortran native
character kind.

.. code-block:: yaml

        f_module:
          iso_c_binding:
          - C_SIZE_T
          "{f_module_name}":
          - "{f_kind}"

            
f_temps
^^^^^^^

A list of suffixes for temporary variable names.

.. code-block:: yaml

    f_temps:
    - len

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

See :ref:`HelpersAnchor` for a description of helper functions.

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

notimplemented
^^^^^^^^^^^^^^

If True the statement is not implemented.
The generated function will have ``#if 0`` surrounding the
wrapper.

This is a way to avoid generating code which will not compile when
the notimplemented wrapper is not needed. For example, the C wrapper
for a C++ function when only the C bufferify wrapper is needed for
Fortran.  The statements should eventually be completed to wrap the
function properly.

Fortran Mixins
--------------

Shroud provides several mixins that provide some common functionality.

.. literalinclude:: ../shroud/fc-statements.json
   :language: json
   :start-after: "sphinx-start-after": "f_mixin_declare-fortran-arg"
   :end-before: "sphinx-end-before": "f_mixin_declare-fortran-arg"

.. literalinclude:: ../shroud/fc-statements.json
   :language: json
   :start-after: "sphinx-start-after": "f_mixin_declare-interface-arg"
   :end-before: "sphinx-end-before": "f_mixin_declare-interface-arg"

