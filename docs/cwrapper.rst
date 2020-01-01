.. Copyright (c) 2017-2019, Lawrence Livermore National Security, LLC and
   other Shroud Project Developers.
   See the top-level COPYRIGHT file for details.

   SPDX-License-Identifier: (BSD-3-Clause)

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

To help control the scope of C names, all externals add a prefix.
It defaults to the first three letters of the
**library** but may be changed by setting the format **C_prefix**:

.. code-block:: yaml

    format:
      C_prefix: NEW_

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

**C_return_code** can be set from the YAML file to override the return value:

.. code-block:: yaml

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

See also *cxx_header*.


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
Note the use of *stdlib* which adds ``std::`` with *language=c++*:

.. code-block:: yaml

    c_header='<stdlib.h>',
    cxx_header='<cstdlib>',
    pre_call=[
        'char * {cxx_var} = (char *) {stdlib}malloc({c_var_len} + 1);',
    ],

See also *c_header*.

  
Predefined types
----------------


Int
^^^

A C ``int`` is represented as:

.. code-block:: yaml

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
to generated *C_memory_dtor_function* used to destroy the memory:

.. code-block:: c++

    struct s_{C_capsule_data_type} {
        void *addr;     /* address of C++ memory */
        int idtor;      /* index of destructor */
    };
    typedef struct s_{C_capsule_data_type} {C_capsule_data_type};

In addition, an identical struct is created for each class.  Having a
unique struct and typedef for each class add a measure of type safety
to the C wrapper:

.. code-block:: c++

    struct s_{C_type_name} {
        void *addr;   /* address of C++ memory */
        int idtor;    /* index of destructor */
    };
    typedef struct s_{C_type_name} {C_type_name};


``idtor`` is the index of the destructor code.  It is used
with memory managerment and discussed in :ref:`MemoryManagementAnchor`.

The C wrapper for a function which returns a class instance will 
return a *C_capsule_data_type* by value.  Functions which take 
a class instance will receive a pointer to a *C_capsule_data_type*.
