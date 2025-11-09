.. Copyright Shroud Project Developers. See LICENSE file for details.

   SPDX-License-Identifier: (BSD-3-Clause)

.. _PointersAndArraysAnchor:

Pointers and Arrays
===================

Shroud will create code to map between C and Fortran pointers.  The
*interoperability with C* features of Fortran 2003 and the
call-by-reference feature of Fortran provides most of the features
necessary to pass arrays to C++ libraries. Shroud can also provide
additional semantic information.  Adding the *rank* attribute
will declare the argument as an assumed-shape array with the given
rank: **+rank(2)** creates ``arg(:,:)``.  The **+dimension(n)** attribute
will instead give an explicit dimension: ``+dimension(10,20)`` creates
``arg(10,20)``.

Using **+dimension** on **+intent(in)** arguments will use the dimension
shape in the Fortran wrapper instead of assumed-shape. This adds some
additional safety since many compiler will warn if the actual argument
is too small.  This is useful when the C++ function has an defined shape.
For example, it expects a pointer to 16 elements.
The Fortran wrapper will pass a pointer to contiguous memory with
no explicit shape information.

.. shape must be known on entry to the routine.

When a function returns a pointer, the default behavior of Shroud
is to convert it into a Fortran variable with the ``POINTER`` attribute
using ``c_f_pointer``. This can be made explicit by adding
**+deref(pointer)** to the function declaration in the YAML file.
For example, ``int *getData(void) +deref(pointer)`` creates the Fortran
function interface

.. code-block:: fortran

    function get_data() result(rv)
        integer(C_INT), pointer :: rv
    end function get_data

The result of the the Fortran function directly accesses the memory
returned from the C++ library.

An array can be returned by adding the **+dimension** attribute to
the function.  The dimension expression will be used to provide the
``shape`` argument to ``c_f_pointer``.  The arguments to **+dimension**
are C++ expressions which are evaluated after the C++ function is
called and can be the name of another argument to the function or a call
another C++ function.

As a simple example, this declaration returns a
pointer to a constant sized array.

.. code-block:: yaml

    - decl: int *returnIntPtrToFixedArray(void) +dimension(10)

Example :ref:`returnIntPtrToFixedArray <example_returnIntPtrToFixedArray>`
shows the generated code.

If the dimension is unknown when the function returns, a ``type(C_PTR)``
can be returned with **+deref(raw)**.  This will allow the user
to call ``c_f_pointer`` once the shape is known.
Instead of a Fortran pointer to a scalar, a scalar can be returned
by adding **+deref(scalar)**.

A common idiom for C++ is to return pointers to memory via arguments.
This would be declared as ``int **arg +intent(out)``.  By default,
Shroud treats the argument similar to a function which returns a
pointer: it adds the *deref(pointer)* attribute to treats it as a
``POINTER`` to a scalar.  The **+dimension** attribute can be used to
create an array similar to a function result.
If the **+deref(allocatable)** attribute is added, then a Fortran array
will be allocated to the size of **+dimension** attribute and the
argument will be copied into the Fortran memory.

.. If *owner(caller)*, then the memory will be released.
   The Fortran ``ALLOCATABLE`` array will need to be released by the user.

A function which returns multiple layers of indirection uses
*deref(raw)* and will return a ``type(C_PTR)``.  This is also true for
function arguments beyond ``int **arg +intent(out)``.
This pointer can represent non-contiguous memory and Shroud
has no way to know the extend of each pointer in the array.

The default behavior of Shroud for *intent(out)* and *intent(inout)*
arguments can be modifed by setting options **F_deref_arg_array**,
**F_deref_arg_character**, **F_deref_arg_implied_array**,
**F_deref_arg_scalar**.  For function results the options are
**F_deref_func_array**, **F_deref_func_character**,
**F_deref_func_implied_array** ** **F_deref_func_scalar**.

A special case is provided for arrays of `NULL` terminated strings,
``char **``.  While this also represents non-contiguous memory, it is a
common idiom and can be processed since the length of each string can
be found with ``strlen``.
See example :ref:`acceptCharArrayIn <example_acceptCharArrayIn>`.

In Python wrappers, Shroud will allocate **+intent(out)** arguments
before calling the function. This requires the dimension attribute
which defines the shape and must be known before the function is
called.  The argument will then be returned by the function along with
the function result and other **+intent(out)** arguments.  For example,
``int **arg +intent(out)+dimension(n)``.  The value of the **+dimension**
attribute is used to define the shape of the array and must be known
before the library function is called.  The **+dimension** attribute can
include the Fortran intrinsic ``size`` to define the shape in terms of
another array.

.. XXX - If no dimension, return as capsule?

.. Python wrappers add some additional requirements on attributes.
   Python will create NumPy arrays for *intent(out)* arguments but
   require an explicit shape using *dimension* attribute. Fortran passes
   in an argument for *intent(out)* arguments which will be filled by the
   C++ library.  However, Python will need to create the NumPy array
   before calling the C++ function.  For example, using
   ``+intent(out)+rank(1)`` will have problems.

Function results
----------------

``char *`` functions have several options.  By default **+deref**
attribute will be set to *allocatable*.  After the C++ function
returns, a ``CHARACTER`` variable will be allocated and the contents
copied.  This will convert a ``NULL`` terminated string into the
proper length of Fortran variable. *+deref(pointer)* returns a pointer
to the library's memory.

For very long strings or strings with embedded ``NULL``,
**+deref(raw)** will return a ``type(C_PTR)``. It is the caller's
responsiblity to dereference the ``C_PTR``, typically by using the
Fortran intrinsic ``c_f_pointer``.

The default value of the *deref* attribute for ``char *`` and
``std::string`` functions is controlled by the option
**F_deref_func_character**.

When the function has the *+funcarg* attribute, the function result
will be returned in a function argument. Adding the *+deref(copy)*
will use the type ``CHARACTER(*)`` for the argument. The C++ function
return value will be copied into the argument. This avoid any issues
with memory management since the caller provides the memory and works
with any version of Fortran. However, if it is too short the result
will be truncated.
See example :ref:`getConstCharPtrAsCopyArg <example_getConstCharPtrAsCopyArg>`.

.. deref(allocatable) allocate before or after call...

.. string, vector
   Pointers to characters will default to *deref(allocatable)*.
   This is useful for short strings.

.. Python  - option.PY_array_arg but not with deref(scalar).


``void *`` functions return a ``type(C_PTR)`` argument and cannot
have **deref**, **dimension**, or **rank** attributes.
A ``type(C_PTR)`` argument will be passed by value.  For a ``void **`` argument,
the ``type(C_PTR)`` will be passed by reference (the default).  This
will allow the C wrapper to assign a value to the argument.
See example :ref:`passVoidStarStar <example_passVoidStarStar>`.


.. +size
   +len
   +len_trim

.. ------------
   

If the C++ library function can also provide the length of the
pointer, then its possible to return a Fortran ``POINTER`` or
``ALLOCATABLE`` variable.  This allows the caller to directly use the
returned value of the C++ function.  However, there is a price; the
user will have to release the memory if **owner(caller)** is set.  To
accomplish this with ``POINTER`` arguments, an additional argument is
added to the function which contains information about how to delete
the array.  If the argument is declared Fortran ``ALLOCATABLE``, then
the value of the C++ pointer are copied into a newly allocated Fortran
array. The C++ memory is deleted by the wrapper and it is the callers
responsibility to ``deallocate`` the Fortran array. However, Fortran
will release the array automatically under some conditions when the
caller function returns. If **owner(library)** is set, the Fortran
caller never needs to release the memory.

.. XXX - std::vector defaults to deref(allocatable) to copy data out of vector.
   The typemap field implied_array is set for std::vector.
   This causes the option *F_deref_func_implied_array* to determine the default
   *deref* attribute.
   

See :ref:`MemoryManagementAnchor` for details of the implementation.


A void pointer may also be used in a C function when any type may be
passed in.  The attribute **assumedtype** can be used to declare a
Fortran argument as assumed-type: ``type(*)``.

.. code-block:: yaml

    - decl: int passAssumedType(void *arg+assumedtype)

.. code-block:: fortran

        function pass_assumed_type(arg) &
                result(SHT_rv) &
                bind(C, name="passAssumedType")
            use iso_c_binding, only : C_INT, C_PTR
            implicit none
            type(*) :: arg
            integer(C_INT) :: SHT_rv
        end function pass_assumed_type

.. _MemoryManagementAnchor:

Memory Management
-----------------

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

When a library function returns a C++ object such as ``std::string``
or ``std::vector`` by value and the Fortran wrapper is returning a
``POINTER`` via *+deref(pointer)* or uses *+deref(raw)*, the C wrapper
must allocate a new instance. In addition to the ``POINTER``, a
*capsule* variable is added as a argument. The caller is responsible
to release the memory via ``call capsule%delete``.  Otherwise the
memory will leak. The ``FINAL`` subroutine of the capsule will be
called when it goes out of scope, so an explicit call to ``delete``
may not be needed.  If the declaration uses *+deref(allocatable)* or
*+deref(copy)*, the wrapper will release the memory before returning
to the caller. At this point the returned varible is owned by Fortran
and released via ``DEALLOCATE`` or going out of scope.

For shadow classes with a destructor defined, the destructor will 
be used to release the memory.

The *c_statements* may also define a way to destroy memory.
For example, the mixin group *c_mixin_destructor_new-vector*
is used with ``std::vector`` and provides the lines:

.. code-block:: yaml

    destructor_name: std_vector_{cxx_T}
    destructor:
    -  std::vector<{cxx_T}> *cxx_ptr = reinterpret_cast<std::vector<{cxx_T}> *>(ptr);
    -  delete cxx_ptr;

Destructor code can be defined without creating a new statement group
by defining it in the **destructors** section of the YAML file.  Then
use the *+destructor_name* attribute in the declaration.
This allows custom destructor code to be used more easily.

The address of the memory to free will be in the variable
``void *ptr``, which should be referenced in the pattern:

.. code-block:: yaml

    declarations:
    - decl: char *getName() +destructor_name(free_getName)

    destructors:
       free_getName: |
          decref(ptr);

Without any explicit *destructor_name*, ``free`` will be
used to release POD pointers; otherwise, ``delete`` will be used.

.. When to use ``delete[] ptr``?

.. There are variants of a capsule based on option.F_assignment_api

   options.F_capsule_variants is a list of variants.
   
   options which set format fields
     F_capsule_data_type_basic_template
     F_capsule_data_type_swig_template
     F_capsule_data_type_rca_template
   These are used by helpers which are evaluted outside the context of a
   class.
   basis does not supply a suffix to make current generated code to be
   unchanged. The other add swig and rca as a suffix.
   

   fmt.F_capsule_data_type set based on options.F_assignment_api
   

C and Fortran
-------------

.. XXX They can be set from the template *F_capsule_data_type_class_template*.
   Need C template too.

Fortran keeps track of C++ objects with the struct
**C_capsule_data_type** and the ``bind(C)`` equivalent
**F_capsule_data_type**. Their names in the format dictionary default to
``{C_prefix}SHROUD_capsule_data`` and ``{C_prefix}SHROUD_capsule_data``.
In the Tutorial these types are defined in :file:`typesTutorial.h` as:

.. literalinclude:: ../regression/reference/classes/typesclasses.h
   :language: c++
   :start-after: start C capsule CLA_Class1
   :end-before: end C capsule CLA_Class1

And :file:`wrapftutorial.f`:

.. literalinclude:: ../regression/reference/classes/wrapfclasses.f
   :language: fortran
   :start-after: start helper capsule_data
   :end-before: end helper capsule_data
   :dedent: 4

*addr* is the address of the C or C++ variable, such as a ``char *``
or ``std::string *``.  *idtor* is a Shroud generated index of the
destructor code defined by *destructor_name* in the statement group
or the *destructor_name* attribute.
These code segments are collected and written to function
*C_memory_dtor_function*.  A value of 0 indicated the memory will not
be released and is used with the **owner(library)** attribute.
*cmemflags* contains bit flags to set pointer properties.

Each class creates its own capsule struct for the C wrapper.
This is to provide a measure of type safety in the C API.
All Fortran classes use the same derived type since the
user does not directly access the derived type.

A typical destructor function would look like:

.. literalinclude:: ../regression/reference/tutorial/utilTutorial.cpp
   :language: c++
   :start-after: start release allocated memory
   :end-before: end release allocated memory

Character and Arrays
^^^^^^^^^^^^^^^^^^^^

.. The option **F_deref_func_character** decides the default value of the *deref* attribute.

In order to create an allocatable copy of a C++ pointer, an additional
structure is involved.  For example, ``getConstStringPtrAlloc``
returns a pointer to a new string. From :file:`strings.yaml`:

.. code-block:: yaml

    declarations:
    - decl: const std::string *getConstStringPtrAlloc() +owner(library)

The C wrapper calls the function and saves the result along with
metadata consisting of the address of the data within the
``std::string`` and its length.  The Fortran wrappers allocates its
return value to the proper length, then copies the data from the C++
variable and deletes it.

The metadata for variables are saved in the C struct **C_array_type**
and the ``bind(C)`` equivalent **F_array_type**.:

.. literalinclude:: ../regression/reference/memdoc/typesmemdoc.h
   :language: c++
   :start-after: start helper array_context
   :end-before: end helper array_context

The union for ``addr`` makes some assignments easier by removing
the need for casts and also aids debugging.
The union is replaced with a single ``type(C_PTR)`` for Fortran:

.. literalinclude:: ../regression/reference/memdoc/wrapfmemdoc.f
   :language: fortran
   :start-after: start helper array_context
   :end-before: end helper array_context
   :dedent: 4

The C wrapper does not return a ``std::string`` pointer.  Instead it
passes in a **C_array_type** pointer as an argument.  It calls
``getConstStringPtrAlloc``, saves the results and metadata into the
argument.  This allows it to be easily accessed from Fortran.
Since the attribute is **owner(library)**, ``cxx.idtor`` is set to ``0``
to avoid deallocating the memory.

.. literalinclude:: ../regression/reference/memdoc/wrapmemdoc.cpp
   :language: c++
   :start-after: start STR_getConstStringPtrAlloc_bufferify
   :end-before: end STR_getConstStringPtrAlloc_bufferify

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
          with a single call by using the *further interoperability
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



.. rubric:: Footnotes

.. [blog1] `<http://blog.enthought.com/python/numpy-arrays-with-pre-allocated-memory>`_

.. [blog2] `<http://blog.enthought.com/python/numpy/simplified-creation-of-numpy-arrays-from-pre-allocated-memory>`_
