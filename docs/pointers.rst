.. Copyright (c) 2017-2020, Lawrence Livermore National Security, LLC and
   other Shroud Project Developers.
   See the top-level COPYRIGHT file for details.

   SPDX-License-Identifier: (BSD-3-Clause)

.. _MemoryManagementAnchor:


Memory Management
=================

Shroud will maintain ownership of memory via the **owner** attribute.
It uses the value of the attribute to decided when to release memory.

Use **owner(library)** when the library owns the memory and the user
should not release it.  For example, this is used when a function
returns ``const std::string &`` for a reference to a string which is
maintained by the library.  Fortran and Python will both get the
reference, copy the contents into their own variable (Fortran
``CHARACTER`` or Python ``str``), then return without releasing any
memory.  This is the default behavior.

Use **owner(caller)** when the library allocates new memory which is
returned to the caller.  The caller is then responsible to release the
memory.  Fortran and Python can both hold on to the memory and then
provide ways to release it using a C++ callback when it is no longer
needed.

For shadow classes with a destructor defined, the destructor will 
be used to release the memory.

The *c_statements* may also define a way to destroy memory.
For example, ``std::vector`` provides the lines:

.. code-block:: yaml

    destructor_name: std_vector_{cxx_T}
    destructor:
    -  std::vector<{cxx_T}> *cxx_ptr = reinterpret_cast<std::vector<{cxx_T}> *>(ptr);
    -  delete cxx_ptr;

Patterns can be used to provide code to free memory for a wrapped
function.  The address of the memory to free will be in the variable
``void *ptr``, which should be referenced in the pattern:

.. code-block:: yaml

    declarations:
    - decl: char * getName() +free_pattern(free_getName)

    patterns:
       free_getName: |
          decref(ptr);

Without any explicit *destructor_name* or pattern, ``free`` will be
used to release POD pointers; otherwise, ``delete`` will be used.

.. When to use ``delete[] ptr``?

C and Fortran
-------------

.. XXX They can be set from the template *F_capsule_data_type_class_template*.
   Need C template too.

Fortran keeps track of C++ objects with the struct
**C_capsule_data_type** and the ``bind(C)`` equivalent
**F_capsule_data_type**. Their names default to
``{C_prefix}SHROUD_capsule_data`` and ``SHROUD_{F_name_scope}capsule``.
In the Tutorial these types are defined in :file:`typesTutorial.h` as:

.. literalinclude:: ../regression/reference/classes/typesclasses.h
   :language: c++
   :start-after: start struct CLA_Class1
   :end-before: end struct CLA_Class1

And :file:`wrapftutorial.f`:

.. literalinclude:: ../regression/reference/classes/wrapfclasses.f
   :language: fortran
   :start-after: start derived-type SHROUD_class1_capsule
   :end-before: end derived-type SHROUD_class1_capsule
   :dedent: 4

*addr* is the address of the C or C++ variable, such as a ``char *``
or ``std::string *``.  *idtor* is a Shroud generated index of the
destructor code defined by *destructor_name* or the *free_pattern* attribute.
These code segments are collected and written to function
*C_memory_dtor_function*.  A value of 0 indicated the memory will not
be released and is used with the **owner(library)** attribute. A
typical function would look like:

.. literalinclude:: ../regression/reference/tutorial/wrapTutorial.cpp
   :language: c++
   :start-after: start release allocated memory
   :end-before: end release allocated memory

Character and Arrays
^^^^^^^^^^^^^^^^^^^^

In order to create an allocatable copy of a C++ pointer, an additional
structure is involved.  For example, ``getConstStringPtrAlloc``
returns a pointer to a new string. From :file:`strings.yaml`:

.. code-block:: yaml

    declarations:
    - decl: const std::string * getConstStringPtrAlloc() +owner(library)

The C wrapper calls the function and saves the result along with
metadata consisting of the address of the data within the
``std::string`` and its length.  The Fortran wrappers allocates its
return value to the proper length, then copies the data from the C++
variable and deletes it.

The metadata for variables are saved in the C struct **C_array_type**
and the ``bind(C)`` equivalent **F_array_type**.:

.. literalinclude:: ../regression/reference/memdoc/typesmemdoc.h
   :language: c++
   :start-after: start array_context
   :end-before: end array_context

The union for ``addr`` makes some assignments easier and also aids debugging.
The union is replaced with a single ``type(C_PTR)`` for Fortran:

.. literalinclude:: ../regression/reference/memdoc/wrapfmemdoc.f
   :language: fortran
   :start-after: start array_context
   :end-before: end array_context
   :dedent: 4

The C wrapper does not return a ``std::string`` pointer.  Instead it
passes in a **C_array_type** pointer as an argument.  It calls
``getConstStringPtrAlloc``, saves the results and metadata into the
argument.  This allows it to be easily accessed from Fortran.
Since the attribute is **owner(library)**, ``cxx.idtor`` is set to ``0``
to avoid deallocating the memory.

.. literalinclude:: ../regression/reference/memdoc/wrapmemdoc.cpp
   :language: c++
   :start-after: start STR_get_const_string_ptr_alloc_bufferify
   :end-before: end STR_get_const_string_ptr_alloc_bufferify

The Fortran wrapper uses the metadata to allocate the return argument
to the correct length:

.. literalinclude:: ../regression/reference/memdoc/wrapfmemdoc.f
   :language: fortran
   :start-after: start get_const_string_ptr_alloc
   :end-before: end get_const_string_ptr_alloc
   :dedent: 4

Finally, the helper function ``SHROUD_copy_string_and_free`` is called
to set the value of the result and possible free memory for
**owner(caller)** or intermediate values:

.. literalinclude:: ../regression/reference/memdoc/utilmemdoc.cpp
   :language: c++
   :start-after: start helper copy_string
   :end-before: end helper copy_string

.. note:: The three steps of call, allocate, copy could be replaced
          with a single call by using the *futher interoperability
          with C* features of Fortran 2018 (a.k.a TS 29113).  This
          feature allows Fortran ``ALLOCATABLE`` variables to be
          allocated by C. However, not all compilers currently support
          that feature.  The current Shroud implementation works with
          Fortran 2003.


Python
------

NumPy arrays control garbage collection of C++ memory by creating 
a ``PyCapsule`` as the base object of NumPy objects.
Once the final reference to the NumPy array is removed, the reference
count on the ``PyCapsule`` is decremented.
When 0, the *destructor* for the capsule is called and releases the C++ memory.
This technique is discussed at [blog1]_ and [blog2]_


Old
---

.. note:: C_finalize is replaced by statement.final


Shroud generated C wrappers do not explicitly delete any memory.
However a destructor may be automatically called for some C++ stl
classes.  For example, a function which returns a ``std::string``
will have its value copied into Fortran memory since the function's
returned object will be destroyed when the C++ wrapper returns.  If a
function returns a ``char *`` value, it will also be copied into Fortran
memory. But if the caller of the C++ function wants to transfer
ownership of the pointer to its caller, the C++ wrapper will leak the
memory.

The **C_finalize** variable may be used to insert code before
returning from the wrapper.  Use **C_finalize_buf** for the buffer
version of wrapped functions.

For example, a function which returns a new string will have to 
``delete`` it before the C wrapper returns:

.. code-block:: c++

    std::string * getConstStringPtrLen()
    {
        std::string * rv = new std::string("getConstStringPtrLen");
        return rv;
    }

Wrapped as:

.. code-block:: yaml

    - decl: const string * getConstStringPtrLen+len=30()
      format:
        C_finalize_buf: delete {cxx_var};

The C buffer version of the wrapper is:

.. code-block:: c++

    void STR_get_const_string_ptr_len_bufferify(char * SHF_rv, int NSHF_rv)
    {
        const std::string * SHCXX_rv = getConstStringPtrLen();
        if (SHCXX_rv->empty()) {
            std::memset(SHF_rv, ' ', NSHF_rv);
        } else {
            ShroudStrCopy(SHF_rv, NSHF_rv, SHCXX_rv->c_str());
        }
        {
            // C_finalize
            delete SHCXX_rv;
        }
        return;
    }

The unbuffer version of the function cannot ``destroy`` the string since
only a pointer to the contents of the string is returned.  It would
leak memory when called:

.. code-block:: c++

    const char * STR_get_const_string_ptr_len()
    {
        const std::string * SHCXX_rv = getConstStringPtrLen();
        const char * SHC_rv = SHCXX_rv->c_str();
        return SHC_rv;
    }

.. note:: Reference counting and garbage collection are still a work in progress




.. rubric:: Footnotes

.. [blog1] `<http://blog.enthought.com/python/numpy-arrays-with-pre-allocated-memory>`_

.. [blog2] `<http://blog.enthought.com/python/numpy/simplified-creation-of-numpy-arrays-from-pre-allocated-memory>`_
