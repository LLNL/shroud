.. Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and
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

Names
-----

Shroud will flatten scoped C++ library names to create the C API.
Since C does not support scopes such as classes and namespaces, a name
such as ``ns1::function`` must be flattened into ``ns1_function`` to
avoid conflict with a similarly named function ``ns2::function``.

Names are also contolled by the **C_api_case** option. It can be set
to *lower*, *upper*, *underscore* or *preserve*. This option is used to set
the format field **C_name_api** which in turn is used in the option
**C_name_template**. The default is *preserve*. This creates a stronger
correlation between the C API and the C++ API.

To further help control the scope of C names, all externals add a prefix.
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
