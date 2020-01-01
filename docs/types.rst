.. Copyright (c) 2017-2019, Lawrence Livermore National Security, LLC and
   other Shroud Project Developers.
   See the top-level COPYRIGHT file for details.

   SPDX-License-Identifier: (BSD-3-Clause)

.. _TypesAnchor:

Types
=====

A typemap is created for each type to describe to Shroud how it should
convert a type between languages for each wrapper.  Native types are
predefined and a Shroud typemap is created for each ``struct`` and
``class`` declaration.

The general form is:

.. code-block:: yaml

    declarations:
    - type: type-name
      fields:
         field1:
         field2:

*type-name* is the name used by C++.  There are some fields which are
used by all wrappers and other fields which are used by language
specific wrappers.

type fields
-----------

These fields are common to all wrapper languages.

base
^^^^

The base type of *type-name*.
This is used to generalize operations for several types.
The base types that Shroud uses are **string**, **vector**, 
or **shadow**.

cpp_if
^^^^^^

A c preprocessor test which is used to conditionally use
other fields of the type such as *c_header* and *cxx_header*:

.. code-block:: yaml

  - type: MPI_Comm
    fields:
      cpp_if: ifdef USE_MPI

flat_name
^^^^^^^^^

A flattened version of **cxx_type** which allows the name to be 
used as a legal identifier in C, Fortran and Python.
By default any scope separators are converted to underscores
i.e. ``internal::Worker`` becomes ``internal_Worker``.
Imbedded blanks are converted to underscores
i.e. ``unsigned int`` becomes ``unsigned_int``.
And template arguments are converted to underscores with the trailing
``>`` being replaced
i.e. ``std::vector<int>`` becomes ``std_vector_int``.

One use of this name is as the **function_suffix** for templated functions.

idtor
^^^^^

Index of ``capsule_data`` destructor in the function
*C_memory_dtor_function*.
This value is computed by Shroud and should not be set.
It can be used when formatting statements as ``{idtor}``.
Defaults to *0* indicating no destructor.

.. format field

result_as_arg
^^^^^^^^^^^^^

Override fields when result should be treated as an argument.
Defaults to *None*.

Statements
----------

Each language also provides a section that is used 
to insert language specific statements into the wrapper.
These are named **c_statements**, **f_statements**, and
**py_statements**.

The are broken down into several resolutions.  The first is the
intent of the argument.  *result* is used as the intent for 
function results.

intent_in
    Code to add for argument with ``intent(IN)``.
    Can be used to convert types or copy-in semantics.
    For example, ``char *`` to ``std::string``.

intent_out
    Code to add after call when ``intent(OUT)``.
    Used to implement copy-out semantics.

intent_inout
    Code to add after call when ``intent(INOUT)``.
    Used to implement copy-out semantics.

result
    Result of function.
    Including when it is passed as an argument, *F_string_result_as_arg*.


Each intent is then broken down into code to be added into
specific sections of the wrapper.  For example, **declaration**,
**pre_call** and **post_call**.

Each statement is formatted using the format dictionary for the argument.
This will define several variables.

c_var
    The C name of the argument.

cxx_var
    Name of the C++ variable.

f_var
    Fortran variable name for argument.

For example:

.. code-block:: yaml

    f_statements:
      intent_in:
      - '{c_var} = {f_var}  ! coerce to C_BOOL'
      intent_out:
      - '{f_var} = {c_var}  ! coerce to logical'

Note that the code lines are quoted since they begin with a curly brace.
Otherwise YAML would interpret them as a dictionary.

See the language specific sections for details.



Numeric Types
--------------

The numeric types usually require no conversion.
In this case the type map is mainly used to generate declaration code 
for wrappers:

.. code-block:: yaml

    type: int
    fields:
        c_type: int 
        cxx_type: int
        f_type: integer(C_INT)
        f_kind: C_INT
        f_module:
            iso_c_binding:
            - C_INT
        f_cast: int({f_var}, C_INT)

One case where a conversion is required is when the Fortran argument
is one type and the C++ argument is another. This may happen when an
overloaded function is generated so that a ``C_INT`` or ``C_LONG``
argument may be passed to a C++ function function expecting a
``long``.  The **f_cast** field is used to convert the argument to the
type expected by the C++ function.


Bool
----


The first thing to notice is that **f_c_type** is defined.  This is
the type used in the Fortran interface for the C wrapper.  The type
is ``logical(C_BOOL)`` while **f_type**, the type of the Fortran
wrapper argument, is ``logical``.

The **f_statements** section describes code to add into the Fortran
wrapper to perform the conversion.  *c_var* and *f_var* default to
the same value as the argument name.  By setting **c_local_var**, a
local variable is generated for the call to the C wrapper.  It will be
named ``SH_{f_var}``.

There is no Fortran intrinsic function to convert between default
``logical`` and ``logical(C_BOOL)``. The **pre_call** and
**post_call** sections will insert an assignment statement to allow
the compiler to do the conversion.


If a function returns a ``bool`` result then a wrapper is always needed
to convert the result.  The **result** section sets **need_wrapper**
to force the wrapper to be created.  By default a function with no
argument would not need a wrapper since there will be no **pre_call**
or **post_call** code blocks.  Only the C interface would be required
since Fortran could call the C function directly.

See example :ref:`checkBool <example_checkBool>`.

Char
----

..  It also helps support ``const`` vs non-``const`` strings.

Any C++ function which has ``char`` or ``std::string`` arguments or
result will create an additional C function which include additional
arguments for the length of the strings.  Most Fortran compiler use
this convention when passing ``CHARACTER`` arguments. Shroud makes
this convention explicit for three reasons:

* It allows an interface to be used.  Functions with an interface will
  not pass the hidden, non-standard length argument, depending on compiler.
* It may pass the result of ``len`` and/or ``len_trim``.
  The convention just passes the length.
* Returning character argument from C to Fortran is non-portable.

Arguments with the *intent(in)* annotation are given the *len_trim*
annotation.  The assumption is that the trailing blanks are not part
of the data but only padding.  Return values and *intent(out)*
arguments add a *len* annotation with the assumption that the wrapper
will copy the result and blank fill the argument so it need to know
the declared length.

The additional function will be named the same as the original
function with the option **C_bufferify_suffix** appended to the end.
The Fortran wrapper will use the original function name, but call the
C function which accepts the length arguments.

The character type maps use the **c_statements** section to define
code which will be inserted into the C wrapper. *intent_in*,
*intent_out*, and *result* subsections add actions for the C wrapper.
*intent_in_buf*, *intent_out_buf*, and *result_buf* are used for
arguments with the *len* and *len_trim* annotations in the additional
C wrapper.

There are occasions when the *bufferify* wrapper is not needed.  For
example, when using ``char *`` to pass a large buffer.  It is better
to just pass the address of the argument instead of creating a copy
and appending a ``NULL``.  The **F_create_bufferify_function** options
can set to *false* to turn off this feature.


Char
^^^^



``Ndest`` is the declared length of argument ``dest`` and ``Lsrc`` is
the trimmed length of argument ``src``.  These generated names must
not conflict with any other arguments.  There are two ways to set the
names.  First by using the options **C_var_len_template** and
**C_var_trim_template**. This can be used to control how the names are
generated for all functions if set globally or just a single function
if set in the function's options.  The other is by explicitly setting
the *len* and *len_trim* annotations which only effect a single
declaration.

The pre_call code creates space for the C strings by allocating
buffers with space for an additional character (the ``NULL``).  The
*intent(in)* string copies the data and adds an explicit terminating
``NULL``.  The function is called then the post_call section copies
the result back into the ``dest`` argument and deletes the scratch
space.  ``ShroudStrCopy`` is a function provided by Shroud which
copies character into the destination up to ``Ndest`` characters, then
blank fills any remaining space.


MPI_Comm
--------

MPI_Comm is provided by Shroud and serves as an example of how to wrap
a non-native type.  MPI provides a Fortran interface and the ability
to convert MPI_comm between Fortran and C. The type map tells Shroud
how to use these routines:

.. code-block:: yaml

        type: MPI_Comm
        fields:
            cxx_type: MPI_Comm
            c_header: mpi.h
            c_type: MPI_Fint
            f_type: integer
            f_kind: C_INT
            f_c_type: integer(C_INT)
            f_c_module:
                iso_c_binding:
                  - C_INT
            cxx_to_c: MPI_Comm_c2f({cxx_var})
            c_to_cxx: MPI_Comm_f2c({c_var})


This mapping makes the assumption that ``integer`` and
``integer(C_INT)`` are the same type.


.. Complex Type
   ------------

Templates
---------

Shroud will wrap templated classes and functions for explicit instantiations.
The template is given as part of the ``decl`` and the instantations are listed in the
``cxx_template`` section:

.. code-block:: yaml

  - decl: |
        template<typename ArgType>
        void Function7(ArgType arg)
    cxx_template:
    - instantiation: <int>
    - instantiation: <double>

``options`` and ``format`` may be provide to control the generated code:

.. code-block:: yaml

  - decl: template<typename T> class vector
    cxx_header: <vector>
    cxx_template:
    - instantiation: <int>
      format:
        C_impl_filename: wrapvectorforint.cpp
      options:
        optblah: two
    - instantiation: <double>

.. from templates.yaml

For a class template, the *class_name* is modified to included the
instantion type.  If only a single template parameter is provided,
then the template argument is used.  For the above example,
*C_impl_filename* will default to ``wrapvector_int.cpp`` but has been
explicitly changed to ``wrapvectorforint.cpp``.


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

.. literalinclude:: ../regression/reference/tutorial/typesTutorial.h
   :language: c++
   :start-after: start struct TUT_Class1
   :end-before: end struct TUT_Class1

And :file:`wrapftutorial.f`:

.. literalinclude:: ../regression/reference/tutorial/wrapftutorial.f
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

.. literalinclude:: ../regression/reference/memdoc/wrapmemdoc.cpp
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
