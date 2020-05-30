.. Copyright (c) 2017-2020, Lawrence Livermore National Security, LLC and
   other Shroud Project Developers.
   See the top-level COPYRIGHT file for details.

   SPDX-License-Identifier: (BSD-3-Clause)

C Statements
============


.. code-block:: text

    extern "C" {

    {C_return_type} {C_name}({C_prototype})
    {
        {pre_call}
        {call_code}
        {post_call_pattern}
        {post_call}
        {final}
        {ret}
    }

c_statements
------------

..        name="c_default",

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

len

    Result of Fortran intrinsic ``LEN`` for string arguments.
    Type ``int``.

len_trim

    Result of Fortran intrinsic ``LEN_TRIM`` for string arguments.
    Type ``int``.

size

    Result of Fortran intrinsic ``SIZE`` for array arguments.
    Type ``long``.

shadow

    Argument will be of type *C_capsule_data_type*.




arg

    default.

shadow
size
capsule
context
len_trim
len

   
buf_extra
^^^^^^^^^


c_header
^^^^^^^^

List of blank delimited header files which will be included by the generated header
for the C wrapper.  These headers must be C only.
For example, ``size_t`` requires stddef.h:

.. code-block:: yaml

    type: size_t
    fields:
        c_type: size_t 
        cxx_type: size_t
        c_header: <stddef.h>


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

cxx_header
^^^^^^^^^^

A blank delimited list of header files which will be added to the C
wrapper implementation.
These headers may include C++ code.

cxx_local_var
^^^^^^^^^^^^^

If a local C++ variable is created for an argument by pre_call,
*cxx_local_var*
indicates if the local variable is a **pointer** or **scalar**.
.. This sets *cxx_var* is set to ``SH_{c_var}``.
This in turns will set the format fields *cxx_member*.
For example, a ``std::string`` argument is created for the C++ function
from the ``char *`` argument passed into the C API wrapper.

c_arg_decl
^^^^^^^^^^

f_arg_decl
^^^^^^^^^^

f_module
^^^^^^^^


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

.. return is a reserved word so it's not possible to do dict(return=[])

return_type
^^^^^^^^^^^

Explicit return type when it is different than the
functions return type.
For example, with shadow types.

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

..        owner="library",
