.. Copyright (c) 2017-2021, Lawrence Livermore National Security, LLC and
   other Shroud Project Developers.
   See the top-level COPYRIGHT file for details.

   SPDX-License-Identifier: (BSD-3-Clause)

C Statements
============

.. note:: Work in progress


.. code-block:: text

    extern "C" {

    {C_return_type} {C_name}({C_prototype})    buf_args
    {
        {pre_call}
        {call_code}   {call}    arg_call
        {post_call_pattern}
        {post_call}
        {final}
        {ret}
    }

A corresponding ``bind(C)`` interface can be created for Fortran.
    
.. code-block:: text

    {F_C_subprogram} {F_C_name}({F_C_arguments}) &
        {F_C_result_clause} &
        bind(C, name="{C_name}")
        f_module / f_module_line
        f_import
        arg_c_decl
    end {F_C_subprogram} {F_C_name}

Where
F_C_clause =
F_C_arguments =   f_c_arg_names
    
c_statements
------------

..        name="c_default",

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


buf_args
^^^^^^^^^

*buf_args* lists the arguments which are used by the C wrapper.
The default is to provide a one-for-one correspondance with the 
arguments of the function which is being wrapped.
However, often an additional function is created which will pass 
additional or different arguments to provide meta-data about the argument.

The Fortran wrapper will call the generated 'bufferified' function
and provide the meta-data to the C wrapper.

arg

    Use the library argument as the wrapper argument.
    This is the default when *buf_args* is not explicit.

arg_decl
    The explicit dummy/prototype declarations will be provided in the fields
    *c_arg_decl* and *f_arg_decl*.
    
capsule

    An argument of type *C_capsule_data_type*/*F_capsule_data_type*.
    It provides a pointer to the C++ memory as well as information
    to release the memory.

    .. XXX need to add helper automatically

context

    An argument of *C_array_type*/*F_array_type*.
    For example, used with ``std::vector`` to hold
    address and size of data contained in the argument
    in a form which may be used directly by Fortran.

    *c_var_context*
    options.C_var_context_template

shadow

    Argument will be of type *C_capsule_data_type*.




arg

    default.

shadow
capsule
context

iface_header
^^^^^^^^^^^^

List of header files which will be included in the generated header
for the C wrapper.  These headers must be C only.
Used for headers needed when *buf_args* contains *arg_decl*.
Can contain headers required for the generated prototypes.

.. note that typemaps will also add c_headers.

impl_header
^^^^^^^^^^^

A list of header files which will be added to the C
wrapper implementation.
These headers may include C++ code.

.. listed in fc_statements as *c_impl_header* and *cxx_impl_header*

c_helper
^^^^^^^^

A blank delimited list of helper functions which will be added to the wrapper file.
The list will be formatted to allow for additional flexibility::

    c_helper: capsule_data_helper vector_context vector_copy_{cxx_T}

These functions are defined in whelper.py.
There is no current way to add additional functions.


c_local_var
^^^^^^^^^^^

If a local C variable is created for the return value by post_call, *c_local_var*
indicates if the local variable is a **pointer** or **scalar**.
For example, when a structure is returned by a C++ function, the C wrapper creates
a local variable which contains a pointer to the C type of the struct.

The local variable can be passed in when buf_args is *shadow*.




If true, generate a local variable using the C declaration for the argument.
This variable can be used by the pre_call and post_call statements.
A single declaration will be added even if with ``intent(inout)``.

cxx_local_var
^^^^^^^^^^^^^

If a local C++ variable is created for an argument by pre_call,
*cxx_local_var*
indicates if the local variable is a **pointer** or **scalar**.
.. This sets *cxx_var* is set to ``SH_{c_var}``.
This in turns will set the format fields *cxx_member*.
For example, a ``std::string`` argument is created for the C++ function
from the ``char *`` argument passed into the C API wrapper.

.. code-block:: yaml

        name="c_inout_string",
        cxx_local_var="scalar",
        pre_call=["{c_const}std::string {cxx_var}({c_var});"],

 Set to **return** when the *c_var* is passed in as an argument and
 a C++ variable must be created.
 Ex ``c_function_shadow``.
 In this case, *cxx_to_c* is defined so a local variable will already
 be created, unless *language=c* in which case *cxx_to_c* is unneeded.

c_arg_decl
^^^^^^^^^^

A list of declarations to append to the prototype in the C wrapper.
Used when *buf_arg* includes "arg_decl".

f_arg_decl
^^^^^^^^^^

A list of dummy argument declarations in the Fortran ``bind(C)``
interface. Used when *buf_arg* includes "arg_decl".  The variable to be
declared is *c_var*.  *f_module* can be used to add ``USE`` statements
needed by the declarations.

.. c_var  c_f_dimension

f_c_arg_names
^^^^^^^^^^^^^

Names of arguments to pass to C function.
Used when *buf_arg* is ``arg_decl``.
Defaults to ``{F_C_var}``.

f_result_decl
^^^^^^^^^^^^^

A list of declarations in the Fortran interface for a function result value.

.. c_var is set to fmt.F_result

f_import
^^^^^^^^

List of names to import into the Fortran interface.
The names will be expanded before being used.

In this example, Shroud creates *F_array_type* derived type in the
module and it is used in the interface.

.. code-block:: yaml

        f_import=["{F_array_type}"],
                

f_module
^^^^^^^^

Fortran modules used in the Fortran interface:

.. code-block:: yaml

        f_module=dict(iso_c_binding=["C_PTR"]),

f_module_line
^^^^^^^^^^^^^

Fortran modules used in the Fortran interface as a single line
which allows format strings to be used.

.. code-block:: yaml

        f_module_line="iso_c_binding:{f_kind}",

The format is::

     module ":" symbol [ "," symbol ]* [ ";" module ":" symbol [ "," symbol ]* ]


arg_call
^^^^^^^^

pre_call
^^^^^^^^

Code used with *intent(in)* arguments to convert from C to C++.

.. the typemap.c_to_cxx field will not be used.

.. * **C_call_code** code used to call the function.
   Constructor and destructor will use ``new`` and ``delete``.

.. * **C_post_call_pattern** code from the *C_error_pattern*.
   Can be used to deal with error values.


call
^^^^

Code to call function.  This is usually generated.
An exception which require explicit call code are constructors
and destructors for shadow types.

post_call
^^^^^^^^^

Code used with *intent(out)* arguments and function results.
Can be used to convert results from C++ to C.

final
^^^^^

Inserted after *post_call* and before *ret*.
Can be used to release intermediate memory in the C wrapper.

.. evaluated in context of fmt_result
       
ret
^^^

Code for return statement.
Usually generated but can be replaced.
For example, with constructors.

Useful to convert a subroutine into a function.
For example, convert a ``void`` function which fills a ``std::vector``
to return the number of items.

.. return is a reserved word so it's not possible to do dict(return=[])

return_type
^^^^^^^^^^^

Explicit return type when it is different than the
functions return type.
For example, with shadow types.

.. code-block:: yaml

      return_type: long
      ret:
      - return Darg->size;

.. from vectors.yaml

*return_type* can also be used to convert a C wrapper into a void
function.  This is useful for functions which return pointers but the
pointer value is assigned to a subroutine argument which holds the
pointer (For example, ``CFI_cdesc_t``).  The ``type(C_PTR)`` which
would be return by the C wrapper is unneeded by the Fortran wrapper.
   

return_cptr
^^^^^^^^^^^

If *true*, the function will return a C pointer. This will be
used by the Fortran interface to declare the function as
``type(C_PTR)``.

 
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

.. c_shadow_scalar_result

Used where the ``new``` operator is part of the generated code.
For example where a class is returned by value or a constructor.
The C wrapper
must explicitly allocate a class instance which will hold the value
from the C++ library function.  The Fortran shadow class must keep
this copy until the shadow class is deleted.

Defaults to *library*.
