.. Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and
   other Shroud Project Developers.
   See the top-level COPYRIGHT file for details.

   SPDX-License-Identifier: (BSD-3-Clause)

C Statements
============

.. note:: Work in progress


.. code-block:: text

    extern "C"
    {C_return_type} {C_name}(c_arg_decl)
    {
        {c_pre_call}
        {c_call_code}
        {c_call}    c_arg_call
        {post_call_pattern}
        {c_post_call}
        {c_final}
        {c_return}
    }

A corresponding ``bind(C)`` interface can be created for Fortran.
    
.. code-block:: text

    {F_C_subprogram} {F_C_name}({i_arg_names}) &
        [result({i_result_var}) & ]
        bind(C, name="{C_name}")
        i_module
        i_import
        i_arg_decl
        i_result_decl
    end {F_C_subprogram} {F_C_name}

.. Typically have different groups for pointer vs reference
   f_out_string* vs f_out_string&
   Since call-by-reference and call-by-value differences.
   Also dereference:  . vs ->
    
Format fields
-------------

* C_prototype -> c_arg_decl
* F_C_clause =
* F_C_arguments     = i_arg_names
* F_C_result_clause = i_result_var

    
Statements
----------

These are listed in the order they appear in the wrapper.

name
^^^^

Must start with a ``c``.

i_arg_names
^^^^^^^^^^^

Names of arguments to pass to C function.
Defaults to ``{F_C_var}``.
An empty list will cause no declaration to be added.

.. note:: *c_arg_decl*, *i_arg_decl*, and *i_arg_names* must all
          exist in a group and have the same number of names.

i_arg_decl
^^^^^^^^^^

A list of dummy argument declarations in the Fortran ``bind(C)``
interface. The variable to be
declared is *c_var*.  *i_module* can be used to add ``USE`` statements
needed by the declarations.
An empty list will cause no declaration to be added.

.. note:: *c_arg_decl*, *i_arg_decl*, and *i_arg_names* must all
          exist in a group and have the same number of names.

.. c_var  c_f_dimension

i_result_decl
^^^^^^^^^^^^^

A list of declarations in the Fortran interface for a function result value.

.. c_var is set to fmt.F_result

i_result_var
^^^^^^^^^^^^
   
i_import
^^^^^^^^

List of names to import into the Fortran interface.
The names will be expanded before being used.

In this example, Shroud creates *F_array_type* derived type in the
module and it is used in the interface.

.. code-block:: yaml

        i_import=["{F_array_type}"],
                

i_module
^^^^^^^^

Fortran modules used in the Fortran interface:

.. code-block:: yaml

        i_module:
          iso_c_binding:
          - C_PTR

Fields will be expanded using the format dictionary before being used.
If unset, then *f_module* will be used when creating the interface.
Shroud will insert ``IMPORT`` statements instead of ``USE`` as needed.

c_arg_decl
^^^^^^^^^^

A list of declarations to append to the prototype in the C wrapper.
Defaults to *None* which will cause Shroud to generate an argument from
the wrapped function's argument.
An empty list will cause no declaration to be added.
Functions do not add arguments by default.
A trailing semicolon will be provided.

.. note:: *c_arg_decl*, *i_arg_decl*, and *i_arg_names* must all
          exist in a group and have the same number of names.

c_arg_call
^^^^^^^^^^

Arguments to pass from the C wrapper to the C++ function.

The value of *None* will pass the C argument
to the C++ function.
The argument will be converted from C to C++ where required.

c_pre_call
^^^^^^^^^^

Code used with *intent(in)* arguments to convert from C to C++.

.. the typemap.c_to_cxx field will not be used.

.. * **C_call_code** code used to call the function.
   Constructor and destructor will use ``new`` and ``delete``.

.. * **C_post_call_pattern** code from the *C_error_pattern*.
   Can be used to deal with error values.


c_call
^^^^^^

Code to call function.  This is usually generated.
An exception which require explicit call code are constructors
and destructors for shadow types.

.. sets need_wrapper

c_post_call
^^^^^^^^^^^

Code used with *intent(out)* arguments and function results.
Can be used to convert results from C++ to C.

.. When the length is greater than 0, typemap.cxx_to_c will not be used
   since the conversion is assumed to be in the c_post_call code.


c_final
^^^^^^^

Inserted after *post_call* and before *ret*.
Can be used to release intermediate memory in the C wrapper.

.. evaluated in context of fmt_result
       
c_return
^^^^^^^^

List of code for return statement.
Usually generated but can be replaced.
For example, with constructors.

Useful to convert a subroutine into a function.
For example, convert a ``void`` function which fills a ``std::vector``
to return the number of items.

c_return_type
^^^^^^^^^^^^^

Explicit return type when it is different than the
functions return type.
For example, with shadow types.

.. code-block:: yaml

      c_return_type: long
      c_return:
      - return Darg->size;

.. from vectors.yaml

*return_type* can also be used to convert a C wrapper into a void
function.  This is useful for functions which return pointers but the
pointer value is assigned to a subroutine argument which holds the
pointer (For example, ``CFI_cdesc_t``).  The ``type(C_PTR)`` which
would be return by the C wrapper is unneeded by the Fortran wrapper.

The Fortran wrapper is also changed to call the C wrapper as a subroutine.

.. The field will be expanded so it may be set to "{c_type}";
   however, the Fortran wrapper does not parse the value so
   if it is a pointer, ``int *``, the typemap will not be found.
   It will also be necessary to set i_result_decl.

c_temps
^^^^^^^

A list of suffixes for temporary variable names.

.. code-block:: yaml

    c_temps=["len"]

Create variable names in the format dictionary using
``{fmt.c_temp}{rootname}_{name}``.
For example, argument *foo* creates *SHT_foo_len*.

The format field is named *c_var_{name}*.

This field is also used to create names for the Fortran interface.
In this case the format field is named *i_var_{name}*.

c_local
^^^^^^^

Similar to *temps* but uses ``{fmt.C_local}{rootname}_{name}``.
*temps* is intended for arguments and is typically used in a mixin
group.  *local* is used by group to generate names for local
variables.  This allows creating names without conflicting with
*temps* from a *mixin* group.

The format field is named *c_local_{name}*.
   
c_helper
^^^^^^^^

A list of helper functions which will be added to the wrapper file.
The format dictionary will be applied to the list for additional
flexibility.

.. code-block:: yaml

    c_helper:
    - capsule_data_helper
    - vector_context
    - vector_copy_{cxx_T}

Each helper will add an entry into the format dictionary with
the name of the function or type created by the helper.
The format value is the helper name prefixed by *c_helper_*.
For example, format field *c_helper_capsule_data_helper* may be ``TEM_SHROUD_capsule_data``.

There is no current way to add additional helper functions.

.. These functions are defined in whelper.py.

c_need_wrapper
^^^^^^^^^^^^^^

There are occassions when a C wrapper is not needed when the Fortran
wrapper can call the library function directly. For example, when
language=c or the C++ library function is ``extern "C"``.

*c_need_wrapper* can be set to *True* to force the C wrapper to be
created.  This is useful when the wrapper is modified via other fields
such as *c_return_type*.

iface_header
^^^^^^^^^^^^

List of header files which will be included in the generated header
for the C wrapper.  These headers must be C only and will be
included after ``ifdef __cplusplus``.
Used for headers needed for declarations in *c_arg_decl*.
Can contain headers required for the generated prototypes.

For example, ``ISO_Fortran_binding.h`` is C only.

.. The Cray ftn compiler requires extern "C".

.. note that typemaps will also add c_headers.

impl_header
^^^^^^^^^^^

A list of header files which will be added to the C
wrapper implementation.
These headers may include C++ code.

.. listed in fc_statements as *c_impl_header* and *cxx_impl_header*

destructor_name
^^^^^^^^^^^^^^^

A name for the destructor code in *destructor*.
Must be unique.  May include format strings:

.. code-block:: yaml

    destructor_name: std_vector_{cxx_T}

destructor
^^^^^^^^^^

A list of lines of code used to delete memory. Usually allocated by a *pre_call*
statement.  The code is inserted into *C_memory_dtor_function* which will provide
the address of the memory to destroy in the variable ``void *ptr``.
For example:

.. code-block:: yaml

    destructor:
    -  std::vector<{cxx_T}> *cxx_ptr = reinterpret_cast<std::vector<{cxx_T}> *>(ptr);
    -  delete cxx_ptr;

owner
^^^^^

Set *owner* of the memory.
Similar to attribute *owner*.

.. XXX example in c_function_shadow_scalar

Used where the ``new``` operator is part of the generated code.
For example where a class is returned by value or a constructor.
The C wrapper
must explicitly allocate a class instance which will hold the value
from the C++ library function.  The Fortran shadow class must keep
this copy until the shadow class is deleted.

Defaults to *library*.


lang_c and lang_cxx
^^^^^^^^^^^^^^^^^^^

Language specific versions of each field can be added to these
dictionaries. The version which corresponds to the YAML file
*language* field will be used.

.. code-block:: yaml

        lang_c=dict(
            impl_header=["<stddef.h>"],
        ),
        lang_cxx=dict(
            impl_header=["<cstddef>"],
        ),
                
