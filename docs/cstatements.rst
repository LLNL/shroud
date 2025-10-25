.. Copyright Shroud Project Developers. See LICENSE file for details.

   SPDX-License-Identifier: (BSD-3-Clause)

.. _top_C_Statements:
   
C Statements
============

.. note:: Work in progress

``{}`` denotes a format field. ``[]`` is a statement field.

.. code-block:: text

    extern "C"
    {C_return_type} {C_name}({C_prototype})
    {
        [c_pre_call]
        [c_call]
        [post_call_pattern]
        [c_post_call]
        [c_final]
        [c_return]
    }

A corresponding ``bind(C)`` interface can be created for Fortran.
    
.. code-block:: text

    {i_subprogram} {i_name}({i_dummy_arg}) &
        {i_pure_clause} {i_result_clause} &
        bind(C, name="{C_name}")
        [i_module]
        [i_import]
        [i_dummy_decl]
        [i_result_decl]
    end {i_subprogram} {i_name}

.. Typically have different groups for pointer vs reference
   f_out_string* vs f_out_string&
   Since call-by-reference and call-by-value differences.
   Also dereference:  . vs ->
    
Format fields
-------------

* C_prototype
  Built up from *c_prototype*.

* C_call_function
  Built up from *c_arg_call*.
  Used with statement *c_call*.
  
* i_pure_clause
* i_arguments     = join i_dummy_arg
* i_result_clause = i_result_var

    
Statements
----------

These are listed in the order they appear in the wrapper.

name
^^^^

Must start with a ``c``.

i_dummy_arg
^^^^^^^^^^^

List of dummy argument names for the Fortran interface.
An empty list will cause no declaration to be added.

.. note:: *c_prototype*, *i_dummy_decl*, and *i_dummy_arg* must all
          exist in a group and have the same number of names.

i_dummy_decl
^^^^^^^^^^^^

A list of dummy argument declarations in the Fortran ``bind(C)``
interface. The variable to be
declared is *c_var*.  *i_module* can be used to add ``USE`` statements
needed by the declarations.
An empty list will cause no declaration to be added.

.. note:: *c_prototype*, *i_dummy_decl*, and *i_dummy_arg* must all
          exist in a group and have the same number of names.

.. c_var  c_f_dimension

i_result_decl
^^^^^^^^^^^^^

A list of declarations in the Fortran interface for a function result value.

.. c_var is set to fmt.F_result
.. does not require i_dummy_arg

i_result_var
^^^^^^^^^^^^
   
i_import
^^^^^^^^

List of names to import into the Fortran interface.
The names will be expanded before being used.

In this example, Shroud creates *F_array_type* derived type in the
module and it is used in the interface.

.. code-block:: yaml

        i_import: ["{F_array_type}"]
                

i_module
^^^^^^^^

Fortran modules used in the Fortran interface:

.. code-block:: yaml

        i_module:
          iso_c_binding:
          - C_PTR
          "{f_module_name}":
          - "{f_kind}"

Fields will be expanded using the format dictionary before being used.
If *i_module* is not set, *f_module* will be used when creating the interface.
Shroud will insert ``IMPORT`` statements instead of ``USE`` as needed.

c_prototype
^^^^^^^^^^^

A list of declarations to create the format field *C_prototype*.
An empty list will cause no declaration to be added.
Functions do not add an argument by default.

.. note:: *c_prototype*, *i_dummy_decl*, and *i_dummy_arg* must all
          exist together in a statement group and have the same number of names.

c_arg_call
^^^^^^^^^^

Arguments to pass from the C wrapper to the C++ function.

The value of *None* will pass the C argument
to the C++ function.
The argument will be converted from C to C++ where required.

c_body
^^^^^^

The entire declaration of the function.
Only used with assignment overload.

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

Code to call function.
Typically, for ``void`` functions ``{C_call_function};`` and for other functions
``{gen.cxxresult.c_var} = {C_call_function}``.

An example of explicit *c_call* code are constructors and destructors
for shadow types.

*getter* and *setter* functions will not need this as well
since the wrapper operates directly on the struct and not a function.


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

    c_temps:
    - len

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

*local* format fields are not created for Fortran interfaces which
have no executable code and do not require local variables.

helper
^^^^^^

A list of helper functions which will be added to the wrapper file.
The format dictionary will be applied to the list for additional
flexibility.

.. code-block:: yaml

    helper:
    - capsule_data_helper
    - vector_context
    - vector_copy_{cxx_T}

Each helper will add an entry into the format dictionary with
the name of the function or type created by the helper
defined in the helper's *c_fmtname* field.
The format value is the helper name prefixed by *c_helper_*.
For example, format field *c_helper_capsule_data_helper* may be
``TEM_SHROUD_capsule_data``.

See :ref:`HelpersAnchor` for a description of helper functions.

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
Used for headers needed for declarations in *c_prototype*.
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

destructor_header
^^^^^^^^^^^^^^^^^

A list of header files which will be added to the C++ utility file.
These headers may include C++ code.

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

Defaults to *None*.


lang_c and lang_cxx
^^^^^^^^^^^^^^^^^^^

Language specific versions of each field can be added to these
dictionaries. The version which corresponds to the YAML file
*language* field will be used.

.. code-block:: yaml

        lang_c:
          impl_header:
          - "<stddef.h>"
        lang_cxx:
          impl_header:
          - "<cstddef>"

C Mixins
--------

Shroud provides several mixins that provide some common functionality.

.. literalinclude:: ../shroud/fc-statements.json
   :language: json
   :start-after: "sphinx-start-after": "c_mixin_declare-arg"
   :end-before: "sphinx-end-before": "c_mixin_declare-arg"
