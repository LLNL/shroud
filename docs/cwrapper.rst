.. Copyright (c) 2017-2018, Lawrence Livermore National Security, LLC. 
.. Produced at the Lawrence Livermore National Laboratory 
..
.. LLNL-CODE-738041.
.. All rights reserved. 
..
.. This file is part of Shroud.  For details, see
.. https://github.com/LLNL/shroud. Please also read shroud/LICENSE.
..
.. Redistribution and use in source and binary forms, with or without
.. modification, are permitted provided that the following conditions are
.. met:
..
.. * Redistributions of source code must retain the above copyright
..   notice, this list of conditions and the disclaimer below.
.. 
.. * Redistributions in binary form must reproduce the above copyright
..   notice, this list of conditions and the disclaimer (as noted below)
..   in the documentation and/or other materials provided with the
..   distribution.
..
.. * Neither the name of the LLNS/LLNL nor the names of its contributors
..   may be used to endorse or promote products derived from this
..   software without specific prior written permission.
..
.. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
.. "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
.. LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
.. A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL LAWRENCE
.. LIVERMORE NATIONAL SECURITY, LLC, THE U.S. DEPARTMENT OF ENERGY OR
.. CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
.. EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
.. PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
.. PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
.. LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
.. NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
.. SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
..
.. #######################################################################


C and C++
=========

A C API is created for a C++ library.  Wrapper functions are within an
``extern "C"`` block so they may be called by C or Fortran.  But the
file must be compiled with the C++ compiler since it is wrapping a C++
library.

When wrapping a C library, additional functions may be created which 
pass meta-data arguments.  When called from Fortran, its wrappers will
provide the meta-data.  When called directly by a C application, the
meta-data must be provided by the user.


Wrapper
-------




As each function declaration is parsed a format dictionary is created
with fields to describe the function and its arguments.
The fields are then expanded into the function wrapper.

C wrapper::

    extern "C" {

    {C_return_type} {C_name}({C_prototype})
    {
        {C_code}
    }

    }

The wrapper is within an ``extern "C"`` block so that **C_name** will
not be mangled by the C++ compiler.

**C_return_code** can be set from the YAML file to override the return value::

    -  decl: void vector_string_fill(std::vector< std::string > &arg+intent(out))
       format:
         C_return_type: int
         C_return_code: return SH_arg.size();

The C wrapper (and the Fortran wrapper) will return ``int`` instead of
``void`` using **C_return_code** to compute the value.  In this case,
the wrapper will return the size of the vector.  This is useful since
C and Fortran convert the vector into an array.


.. wrapc.py   Wrapc.write_header

Types
-----

The typemap provides several fields used to convert between C and C++.

type fields
-----------

c_type
^^^^^^

Name of type in C.
Default to *None*.


c_header
^^^^^^^^

Name of C header file required for type.
This file is included in the interface header.
Only used with *language=c*.
Defaults to *None*.


c_to_cxx
^^^^^^^^

Expression to convert from C to C++.
Defaults to *None* which implies *{c_var}*.
i.e. no conversion required.


c_templates
^^^^^^^^^^^

c_statements for cxx_T

A dictionary indexed by type of specialized *c_statements* When an
argument has a *template* field, such as type ``vector<string>``, some
additional specialization of c_statements may be required::

        c_templates:
            string:
               intent_in_buf:
               - code to copy CHARACTER to vector<string>



c_return_code
^^^^^^^^^^^^^

None

c_union
^^^^^^^

None
# Union of C++ and C type (used with structs and complex)

cxx_type
^^^^^^^^

Name of type in C++.
Defaults to *None*.


cxx_to_c
^^^^^^^^

Expression to convert from C++ to C.
Defaults to *None* which implies *{cxx_var}*.
i.e. no conversion required.

cxx_header
^^^^^^^^^^

Name of C++ header file required for implementation.
For example, if cxx_to_c was a function.
Only used with *language=c++*.
Defaults to *None*.

Statements
----------

The *C_code* field has a default value of::

    {C_return_type} {C_name}({C_prototype})
    {
        {C_pre_call}
        {C_call_code}
        {C_post_call_pattern}
        {C_post_call}
        {C_return_code}
    }


buf_args
^^^^^^^^^

*buf_args* lists the arguments which are added to the wrapper.
The default is to provide a one-for-one correspondance with the 
arguments of the function which is being wrapped.
However, often an additional function is created which will pass 
additional arguments to provide meta-data about the argument.

The Fortran wrapper will call the generated 'bufferified' function
and provide the meta-data to the C wrapper.

arg

    Use the library argument as the wrapper argument.
    This is the default when *buf_args* is not explicit.

capsule

    An argument of type *C_capsule_data_type*/*F_capsule_data_type*.
    It provides a pointer to the C++ memory as well as information
    to release the memory.

context

    An argument of *C_context_type*/*F_context_type*.
    For example, used with ``std::vector`` to hold
    address and size of data contained in the argument
    in a form which may be used directly by Fortran.

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



cxx_local_var
^^^^^^^^^^^^^

If a local C++ variable must be created from the C argument, *cxx_local_var*
indicates if the local variable is a **pointer** or **scalar**.
.. This sets *cxx_var* is set to ``SH_{c_var}``.
This in turns will set the format fields *cxx_member*.

c_header
^^^^^^^^

List of blank delimited header files which will be included by the generated header
for the C wrapper.  These headers must be C only.
For example, ``size_t`` requires stddef.h::

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

cxx_header
^^^^^^^^^^

A blank delimited list of header files which will be added to the C wrapper implementation.
These headers may include C++ code.

destructor
^^^^^^^^^^

A list of lines of code used to delete memory. Usually allocated by a *pre_call*
statement.  The code is inserted into *C_memory_dtor_function* which will provide
the address of the memory to destroy in the variable ``void *ptr``.
For example::

    destructor:
    -  std::vector<{cxx_T}> *cxx_ptr = reinterpret_cast<std::vector<{cxx_T}> *>(ptr);
    -  delete cxx_ptr;


destructor_name
^^^^^^^^^^^^^^^

A name for the destructor code in *destructor*.
Must be unique.  May include format strings::

    destructor_name: std_vector_{cxx_T}


pre_call
^^^^^^^^

Code used with *intent(in)* arguments to convert from C to C++.

.. * **C_call_code** code used to call the function.
   Constructor and destructor will use ``new`` and ``delete``.

.. * **C_post_call_pattern** code from the *C_error_pattern*.
   Can be used to deal with error values.

post_call
^^^^^^^^^

Code used with *intent(out)* arguments.
Can be used to convert results from C++ to C.

.. Includes any code from **C_finalize**.

* **C_return_code** returns a value from the wrapper.


Predefined types
----------------


Int
^^^

A C ``int`` is represented as::

    type: int
    fields:
        c_type: int 
        cxx_type: int


Struct Type
-----------

While C++ considers a struct and a class to be similar, Shroud assumes
a struct is intended to be a C compatible data structure.
It has no methods which will cause a v-table to be created.
This will cause an array of structs to be identical in C and C++.

The main use of wrapping a struct for C is to provide access to the name.
If the struct is defined within a ``namespace``, then a C application will be
unable to access the struct.  Shroud creates an identical struct as the
one defined in the YAML file but at the global level.


Class Types
-----------

A C++ class is represented by the *C_capsule_data_type*.  This struct
contains a pointer to the C++ instance allocated and an index passed
to generated *C_memory_dtor_function* used to destroy the memory::

    struct s_{C_capsule_data_type} {
        void *addr;     /* address of C++ memory */
        int idtor;      /* index of destructor */
        int refcount;   /* reference count */
    };
    typedef struct s_{C_capsule_data_type} {C_capsule_data_type};

In addition, an identical struct is created for each class.  Having a
unique struct and typedef for each class add a measure of type safety
to the C wrapper::

    struct s_{C_type_name} {
        void *addr;   /* address of C++ memory */
        int idtor;    /* index of destructor */
        int refcount; /* reference count */
    };
    typedef struct s_{C_type_name} {C_type_name};


The function *C_capsule_data_type* is generated to have a destructor
for each class.  A single function is created to avoid polluting the
global namespace with multiple functions.  It is also easier to
control ownership using *idtor*.  If the wrapped library owns the
memory (i.e. the C wrapper never needs to release the memory) then
*idtor* will be 0::

    void {C_memory_dtor_function}({C_capsule_data_type} *cap, bool gc)
    {
        void *ptr = cap->addr;
        switch (cap->idtor) {
        case 0:
        {
            // Nothing to delete
            break;
        }
        case 1:
        {
            tutorial::Class1 *cxx_ptr = 
                reinterpret_cast<tutorial::Class1 *>(ptr);
            delete cxx_ptr;
            break;
        }
        default:
        {
            // Unexpected case in destructor
            break;
        }
        }
        if (gc) {
            free(cap);
        } else {
            cap->addr = NULL;
            cap->idtor = 0;  // avoid deleting again
        }
    }
