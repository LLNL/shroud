.. Copyright (c) 2017-2020, Lawrence Livermore National Security, LLC and
   other Shroud Project Developers.
   See the top-level COPYRIGHT file for details.

   SPDX-License-Identifier: (BSD-3-Clause)

.. _TypesSandC:

Structs and Classes
===================

    All problems in computer science can be solved by
    another level of indirection.
    --- David Wheeler

Classes are wrapped by a shadow derived-type with methods implemented
as type-bound procedures in Fortran and an extension type in Python.


Class
-----

Each class in the input file will create a struct which acts as a
shadow class for the C++ class.  A pointer to an instance is saved in
the shadow class. This pointer is then passed down to the C++ routines
to be used as the *this* instance.

Using the tutorial as an example, a simple class is defined in the C++
header as:

.. code-block:: c++

    class Class1
    {
    public:
        void Method1() {};
    };

And is wrapped in the YAML as:

.. code-block:: yaml

    declarations:
    - decl: class Class1
      declarations:
      - decl: int Method1()

Fortran
^^^^^^^

The Fortran interface will create two derived types.  The first is
used to interact with the C wrapper and uses ``bind(C)``. The C
wrapper creates a corresponding struct.  It contains a pointer to an
instance of the class and index used to release the instance.
The ``idtor`` argument is described in :ref:`MemoryManagementAnchor`.

:file:`wrapfclasses.f`

.. literalinclude:: ../regression/reference/classes/wrapfclasses.f
   :language: fortran
   :start-after: start helper capsule_data_helper
   :end-before: end helper capsule_data_helper
   :dedent: 4

:file:`typeclasses.h`

.. literalinclude:: ../regression/reference/classes/typesclasses.h
   :language: c
   :start-after: start struct CLA_Class1
   :end-before: end struct CLA_Class1

The capsule is added to the Fortran shadow class.  This derived type
can contain type-bound procedures and may not use the ``bind(C)``
attribute.

.. code-block:: fortran

    type class1
        type(SHROUD_CLA_capsule_data) :: cxxmem
    contains
        procedure :: method1 => class1_method1
    end type class1

A function which returns a class, including constructors, is passed a
pointer to a *F_capsule_data_type*.  The argument's members are filled
in by the function.  The function will return a ``type(C_PTR)`` which
contains the address of the *F_capsule_data_type* argument.  The
interface/prototype for the C wrapper function allows it to be used in
expressions similar to the way that ``strcpy`` returns its destination
argument.

A generic interface with the same name as the class is created to call
the constructors for the class.  The constructor will initialize the
Fortran derived type.

.. code-block:: fortran

    type(class1) var     ! Create Fortran variable.
    var = class1()       ! Allocate C++ class instance.

When the constructor is wrapped the destructor should also be wrapper or
some other method is provided to release the memory.

Some other type-bound precedures are created to allow the user
to get and set the address of the C++ memory directly.
This can be used when the address of the instance is created in some
other manner (perhaps a C++ module in the application) and it
needs to be used in Fortran without being created in Fortran.
There is no way to free this memory and must be released outside of Fortran.

.. XXX unless idtor is set properly

.. code-block:: fortran

    type(class1) var
    type(C_PTR) addr

    addr = var%get_instance()
    ! addr will not be c_associated
    call var%set_instance(caddr)    ! caddr contains address of an instance
   
Two instances of the class can be compared using the ``associated`` method.

.. code-block:: fortran

    type(class1) var1, var2
    var1 = get_class(1)    ! A library function to fetch an instance
    var2 = get_class(2)
    if (var1%associated(var2) then
        print *, "Identical instances"
    endif


A full example is at 
:ref:`Constructor and Destructor <example_constructor_and_destructor>`.

Inheritance is implemented using the ``EXTENDS`` Fortran keyword.
Only single inheritance is supported.

.. code-block:: fortran

    type shape
        type(CLA_SHROUD_capsule_data) :: cxxmem
    contains
        procedure :: get_ivar => shape_get_ivar
    end type shape

    type, extends(shape) :: circle
    end type circle
                
..
 The *f_to_c* field uses the
 generated ``get_instance`` function to return the pointer which will
 be passed to C.

..
 In C an opaque typedef for a struct is created as the type for the C++
 instance pointer.  The *c_to_cxx* and *cxx_to_c* fields casts this
 pointer to C++ and back to C.

..       final! :: {F_name_final}

Python
^^^^^^

An struct is created for each C++ class.

.. literalinclude:: ../regression/reference/classes/pyclassesmodule.hpp
   :language: c
   :start-after: start object PY_Class1
   :end-before: end object PY_Class1

The ``idtor`` argument is used to release memory and described at
:ref:`MemoryManagementAnchor`.  The splicer allows additional fields
to be added by the developer which may be used in function wrappers.

Forward Declaration
^^^^^^^^^^^^^^^^^^^

A class may be forward declared by omitting ``declarations``.
All other fields, such as ``format`` and ``options`` must be provided
on the initial ``decl`` of a Class.
This will define the type and allow it to be used in following declarations.
The class's declarations can be added later:

.. code-block:: yaml

   declarations:
   - decl: class Class1
     options:
        foo: True

   - decl: class Class2
     declarations:
     - decl: void accept1(Class1 & arg1)

   - decl: class Class1
     declarations:
     - decl: void accept2(Class2 & arg2)

.. A class will be forward declared when the ``declarations`` field is
   not provided.  When the class is not defined later in the file, it may
   be necessary to provide the conversion fields to complete the type::
   XXX - define conversion fields

..     declarations:
       - decl: class Class1
         fields:
           c_type: TUT_class1
           f_derived_type: class1
           f_to_c: "{f_var}%get_instance()"
           f_module:
             tutorial_mod:
             - class1

..
 The type map will be written to a file to allow its used by other
 wrapped libraries.  The file is named by the global field
 **YAML_type_filename**. This file will only list some of the fields
 show above with the remainder set to default values by Shroud.


Constructor and Destructor
^^^^^^^^^^^^^^^^^^^^^^^^^^

The constructor and destuctor methods may also be exposed to Fortran.

The class example from the tutorial is:

.. code-block:: yaml

    declarations:
    - decl: class Class1
      declarations:
      - decl: Class1()         +name(new)
        format:
          function_suffix: _default
      - decl: Class1(int flag) +name(new)
        format:
        function_suffix: _flag
      - decl: ~Class1() +name(delete)


The default name of the constructor is ``ctor``.  The name can 
be specified with the **name** attribute.
If the constructor is overloaded, each constructor must be given the
same **name** attribute.
The *function_suffix* must not be explicitly set to blank since the name
is used by the ``generic`` interface.

The constructor and destructor will only be wrapped if explicitly added
to the YAML file to avoid wrapping ``private`` constructors and destructors.

The Fortran wrapped class can be used very similar to its C++ counterpart.

.. code-block:: fortran

    use tutorial_mod
    type(class1) obj
    integer(C_INT) i

    obj = class1_new()
    i = obj%method1()
    call obj%delete

For wrapping details see 
:ref:`Constructor and Destructor <example_constructor_and_destructor>`.

..  chained function calls

Member Variables
^^^^^^^^^^^^^^^^

For each member variable of a C++ class a C and Fortran wrapper
function will be created to get or set the value.  The Python wrapper
will create a descriptor:

.. code-block:: c++

    class Class1
    {
    public:
       int m_flag;
       int m_test;
    };

It is added to the YAML file as:

.. code-block:: yaml

    - decl: class Class1
      declarations:
      - decl: int m_flag +readonly;
      - decl: int m_test +name(test);

The *readonly* attribute will not write the setter function or descriptor.
Python will report:

.. code-block:: python

    >>> obj = tutorial.Class1()
    >>> obj.m_flag =1
    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
    AttributeError: attribute 'm_flag' of 'tutorial.Class1' objects is not writable

The *name* attribute will change the name of generated functions and
descriptors.  This is helpful when using a naming convention like
``m_test`` and you do not want ``m_`` to be used in the wrappers.

For wrapping details see 
:ref:`Getter and Setter <example_getter_and_setter>`.

Struct
------

Shroud supports both structs and classes. But it treats them much
differently.  Whereas in C++ a struct and class are essentially the
same thing, Shroud treats structs as a C style struct.  They do not
have associated methods.  This allows them to be mapped to a Fortran
derived type with the ``bind(C)`` attribute and a Python NumPy array.

A struct is defined in a single ``decl`` in the YAML file.

.. code-block:: yaml

    - decl: struct Cstruct1 {
              int ifield;
              double dfield;
            };

.. _struct_fortran:

Fortran
^^^^^^^

This is translated directly into a Fortran derived type with the
``bind(C)`` attribute.

.. literalinclude:: ../regression/reference/struct-c/wrapfstruct.f
   :language: fortran
   :start-after: start derived-type cstruct1
   :end-before: end derived-type cstruct1
   :dedent: 4

All creation and access of members can be done using Fortran.

.. code-block:: fortran

    type(cstruct1) st(2)

    st(1)%ifield = 1_C_INT
    st(1)%dfield = 1.5_C_DOUBLE
    st(2)%ifield = 2_C_INT
    st(2)%dfield = 2.6_C_DOUBLE

The option *wrap_struct_as* can be set to *class* to create a shadow type
identical to how classes are wrapped.  This is useful when wrapping C
code which does not support ``class`` directly.
*wrap_struct_as* defaults to *struct* which will create a derived type.
*wrap_class_as* also exists and defaults to *class*.

Python
^^^^^^

Python can treat a struct in several different ways by setting option
*PY_struct_arg*.
First, treat it the same as a class.  An extension type is created with
descriptors for the field methods. Second, as a numpy descriptor.
This allows an array of structs to be used easily.
Finally, as a tuple of Python types.

.. PY_struct_arg *class*, *numpy*, *list*

When treated as a class, a constructor is created which will
create an instance of the class.  This is similar to the
default constructor for structs in C++ but will also work
with a C struct.

.. code-block:: python

    import cstruct
    a = cstruct.Cstruct1(1, 2.5)
    a = cstruct.Cstruct1()

.. regression/run/struct-c/python/test.py

When treated as a NumPy array no memory will be copied since the
NumPy array contains a pointer to the C++ memory.

.. code-block:: python

    import cstruct
    dt = cstruct.Cstruct1_dtype
    a = np.array([(1, 1.5), (2, 2.6)], dtype=dt) 

The descriptor is created in the wrapper
:ref:`NumPy Struct Descriptor <pyexample_Numpy Struct Descriptor>`.


.. XXX array members in struct
   char name[20]    s.name = None   will add set to '\0'
   int  count[10]   s.count = 0     will broadcast
