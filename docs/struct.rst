.. Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and
   other Shroud Project Developers.
   See the top-level COPYRIGHT file for details.

   SPDX-License-Identifier: (BSD-3-Clause)

.. _types-s-and-c:

Structs and Classes
===================

    All problems in computer science can be solved by
    another level of indirection.
    --- David Wheeler

While structs and classes are very similar in C++, Shroud wraps them
much differently.  Shroud treats structs as they are in C and creates
a corresponding derived type for the struct. Shroud wraps classes by
creating a shadow class which holds a pointer to the instance
then uses Fortran type bound procedures to implement methods.


Class
-----

Classes are wrapped by creating a *shadow class* for the C++ class.
A pointer to an instances is saved along with a memory management flag.

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
      - decl: Class1()
        format:
          function_suffix: _default
      - decl: Class1(int flag)
        format:
          function_suffix: _flag
      - decl: ~Class1()        +name(delete)

While C++ will provide a default constructor and destructor, they must
be listed explicitly to wrap them.  They are not assumed since they
may be private.  The default name of the constructor is ``ctor``.  The
name can be specified with the *name* attribute.  If the constructor
is overloaded, each constructor must be given the same *name*
attribute.

The *function_suffix* is added to distinguish overloaded constructors.
The default is to have a sequential numeric suffix.  The
*function_suffix* must not be explicitly set to blank since the name
is used by the Fortran ``generic`` interface.

If no constructor is wrapped, then some other factory method
should be available to create instances of the class. There is no way
to create it directly from Fortran.

When the constructor is wrapped the destructor should also be wrapper
or some other method should be wrapped to release the memory.

C
^

Each class in the YAML file will create a struct in the C wrapper.
All of these structs are identical but are named after the class.
This is intended to provide some level of type safety by making it
harder to accidently use the wrong class with a method.
Shroud refers to this as a capsule.

.. literalinclude:: ../regression/reference/classes/typesclasses.h
   :language: c
   :start-after: start C capsule CLA_Class1
   :end-before: end C capsule CLA_Class1

The C wrapper will extract the address of the instance then call the
method.

.. literalinclude:: ../regression/reference/classes/wrapClass1.cpp
   :language: c++
   :start-after: start CLA_Class1_Method1
   :end-before: end CLA_Class1_Method1
                
All constructors are very similar. They call the C++ constructor then
saves the pointer to the instance.  The *idtor* field is the index of
the destructor maintained by Shroud to destroy the instance.

.. literalinclude:: ../regression/reference/classes/wrapClass1.cpp
   :language: c++
   :start-after: start CLA_Class1_ctor_flag
   :end-before: end CLA_Class1_ctor_flag

Finally the wrapper for the destructor.
The *addr* field is cleared to avoid a dangling pointer.

.. literalinclude:: ../regression/reference/classes/wrapClass1.cpp
   :language: c++
   :start-after: start CLA_Class1_delete
   :end-before: end CLA_Class1_delete

A function which returns a class, including constructors, is passed a
pointer to a *F_capsule_data_type*.  The argument's members are filled
in by the function.  The function will return a ``type(C_PTR)`` which
contains the address of the *F_capsule_data_type* argument.  The
prototype for the C wrapper function allows it to be used in
expressions similar to the way that ``strcpy`` returns its destination
argument.  The option *C_shadow_result* can be set to *False* to
change the function to return `void` instead.
The Fortran wrapper API will be uneffected.

C++ functions which return `const` pointers will not create a `const`
C wrapper. This is because the C wapper will return a pointer to the
capsule and not the instance.

Fortran
^^^^^^^

.. name of derived type

The Fortran wrapper uses the object-oriented features added in
Fortran 2003.  There is one derived type for the library which is
used as the capsule.  This derived type uses ``bind(C)`` since it is
passed to the C wrapper. Each class uses the same capsule derived type
since it is considered an implementation detail and the user should
not access it.  Then each class creates an additional derived type as
the *shadow* class which contains a capsule and has type-bound
procedures associated with it.

.. :file:`wrapfclasses.f`

.. literalinclude:: ../regression/reference/classes/wrapfclasses.f
   :language: fortran
   :start-after: start helper capsule_data_helper
   :end-before: end helper capsule_data_helper
   :dedent: 4

The capsule is added to the Fortran shadow class.  This derived type
can contain type-bound procedures and may not use the ``bind(C)``
attribute.

.. code-block:: fortran

    type class1
        type(SHROUD_CLA_capsule_data) :: cxxmem
    contains
        procedure :: delete => class1_delete
        procedure :: method1 => class1_method1
    end type class1

The wrapper for the method passes the object as the first argument.
The argument uses the format field *F_this* to name the variable and
defaults to ``obj``.  It can be renamed if it conflicts with
another argument.

.. literalinclude:: ../regression/reference/classes/wrapfclasses.f
   :language: fortran
   :start-after: start class1_method1
   :end-before: end class1_method1
   :dedent: 4

A generic interface with the same name as the class is created to call
the constructors for the class.  The constructor will initialize the
Fortran shadow class.

.. literalinclude:: ../regression/reference/classes/wrapfclasses.f
   :language: fortran
   :start-after: start generic interface class1
   :end-before: end generic interface class1
   :dedent: 4

The Fortran wrapped class can be used very similar to its C++ counterpart.
            
.. code-block:: fortran

    use classes_mod
    type(class1) var     ! Create Fortran variable.
    integer(C_INT) i
    var = class1()       ! Allocate C++ class instance.
    i = var%method1()
    call var%delete

Some additional type-bound procedures are created to allow the user to
get and set the address of the C++ memory directly.  This can be used
when the address of the instance is created in some other manner and
it needs to be used in Fortran.  There is no way to free this memory
and it must be released outside of Fortran.

.. XXX unless idtor is set properly

For example, a C++ function creates an instance then passes the
address of it to Fortran function ``worker``. The shadow class is
initialized with the address and can then be used in an object-oriented
fashion:

.. code-block:: fortran

    subroutine worker(addr) bind(C)
    use classes_mod
    type(C_PTR), intent(IN) :: addr
    type(class1) var
    integer(C_INT) i

    call var%set_instance(addr)
    i = var%method1()
   
Two instances of the class can be compared using the ``associated`` method.

.. code-block:: fortran

    type(class1) var1, var2
    var1 = get_class(1)    ! A library function to fetch an instance
    var2 = get_class(2)
    if (var1%associated(var2) then
        print *, "Identical instances"
    endif

These functions names are controlled by format fields *F_name_associated*,
*F_name_instance_get* and *F_name_instance_set*.
If the names are blank, the functions will not be created.

The `.eq.` operator is also defined.

.. F_name_assign  F_name_final

A full example is at 
:ref:`Constructor and Destructor <example_constructor_and_destructor>`.

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

An PyObject is created for each C++ class.
It constains the same values as the capsule.

.. literalinclude:: ../regression/reference/classes/pyclassesmodule.hpp
   :language: c
   :start-after: start object PY_Class1
   :end-before: end object PY_Class1

The ``idtor`` argument is used to release memory and described at
:ref:`MemoryManagementAnchor`.  The splicer allows additional fields
to be added by the developer which may be used in function wrappers.

Additional fields can be added to the splicer for custom behavior.

Chained functions
-----------------

C++ allows class methods to be chained by returning the ``this`` argument.
Several functions can be called in succession on the same object.

.. code-block:: c++

    auto var = Class1()->returnThis()

The *return_this* field indicates that the function may be chained
so the wrapper can generate appropriate code.
    
.. literalinclude:: ../regression/input/classes.yaml
   :language: yaml
   :start-after: start returnThis
   :end-before: end returnThis

C
^

The C wrapper returns ``void`` instead of a pointer to the *this* argument.

.. literalinclude:: ../regression/reference/classes/wrapClass1.cpp
   :language: c++
   :start-after: start CLA_Class1_returnThis
   :end-before: end CLA_Class1_returnThis

                
Fortran
^^^^^^^
         
Fortran does not permit his behavior.
The function is treated as a ``subroutine``.

.. literalinclude:: ../regression/reference/classes/wrapfclasses.f
   :language: fortran
   :start-after: start class1_return_this
   :end-before: end class1_return_this
   :dedent: 4

The chaining must be done as a sequence of calls.

.. code-block:: fortran

   use classes_mod
   type(class1) var

   var = class1()
   call var%return_this()


Class static methods
--------------------

To wrap the method:

.. code-block:: c++

    class Singleton {
        static Singleton& getReference();
    };

Use the YAML input:

.. code-block:: yaml

    - decl: class Singleton
      declarations:
      - decl: static Singleton& getReference()

Fortran
^^^^^^^

Class static methods are supported using the ``NOPASS`` keyword in Fortran.

.. literalinclude:: ../regression/reference/classes/wrapfclasses.f
   :language: fortran
   :start-after: start derived-type singleton
   :end-before: end derived-type singleton
   :dedent: 4

Called from Fortran as:

.. code-block:: fortran

    type(singleton) obj0
    obj0 = obj0%get_reference()

Note that *obj0* is not assigned a value before the function ``get_reference`` is called.

.. _struct_class_inheritance:
            
Class Inheritance
-----------------

Class inheritance is supported.
Note that the subclass declaration uses a colon and must be quoted. Otherwise
YAML will treat it as another mapping entry.

.. code-block:: yaml

    - decl: class Shape
      declarations:
      - decl: Shape()
      - decl: int get_ivar() const

    - decl: "class Circle : public Shape"
      declarations:
      - decl: Circle()


Fortran
^^^^^^^

Inheritance is implemented using the ``EXTENDS`` Fortran
keyword.  Only single inheritance is supported.

.. code-block:: fortran

    type shape
        type(CLA_SHROUD_capsule_data) :: cxxmem
    contains
        procedure :: get_ivar => shape_get_ivar
    end type shape

    type, extends(shape) :: circle
    end type circle

Python
^^^^^^

Python uses the ``PyTypeObject.tp_base`` field.

Forward Declaration
-------------------

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



..  chained function calls

Member Variables
----------------

For each member variable of a C++ class a C and Fortran wrapper
function will be created to get or set the value.  The Python wrapper
will create a descriptor. It is not necessary to list all members of
the class, only the one which are to be exposed in the wrapper.
``private`` members cannot be wrapped.

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

The *readonly* attribute will not create the setter function or descriptor.
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

The names of these functions are controlled by the options
*SH_class_getter_template* and *SH_class_setter_template*.
They are added as additional methods on the class.

For wrapping details see 
:ref:`Getter and Setter <example_getter_and_setter>`.

The getter and setter for a member which is a pointer to a native type
can use a Fortran pointer if the member is given the *dimension*
attribute.

.. code-block :: yaml

    - decl: class PointerData
      declarations:
      - decl: int nitems;
      - decl: int *items  +dimension(nitems);

Notice that the *dimension* uses another field in the class.
This will create a getter which can be called from Fortran.
Likewise, the setter will require an argument of the same rank as
the *dimension* attribute.

.. code-block :: fortran

    type(PointerData) var
    integer(C_INT) :: nitems
    integer(C_INT), pointer :: items(:)
    integer(C_INT) :: updated(10)

    var = PointerData()
    nitems = var%get_nitems()
    items => var%get_items()

    call var%set_items(updated)
    var%nitems = size(updated)    ! keep nitems and items consistent

The user must be consistent in the use of the getter and setter.  For
example, ``item`` is ``nitems`` long, but if the setter assigns an
array which is shorter, the next call to the getter will still create
a Fortran pointer which is ``nitems`` long.

Another point to note is that the variable ``updated`` should have the
``TARGET`` attribute since we're saving its address in ``var``.

.. XXX array members in struct
   char name[20]    s.name = None   will add set to '\0'
   int  count[10]   s.count = 0     will broadcast

Struct
------

Shroud supports both structs and classes. But it treats them much
differently.  Whereas in C++ a struct and class are essentially the
same thing, Shroud treats structs as a C style struct.  They do not
have associated methods.  This allows them to be mapped to a Fortran
derived type with the ``bind(C)`` attribute and a Python NumPy array.

.. A struct is defined the same as a class with a *declarations* field
   for struct members.
   In addition, a struct can be defined in a single ``decl`` in the YAML file.

For wrapping purposes, a struct is a C++ class without a vtable. It
will contain POD types.  Unlike classes where all member variables do
not need to be wrapped, a struct should be fully defined. This is
necessary to allow an array of structs to be created in the wrapper
language then passed to C++.
   
A struct is defined in the yaml file as:


.. code-block:: yaml

    - decl: struct Cstruct1
      declarations:
      - decl: int ifield;
      - decl: double dfield;

It can also be defined as one decl entry:

.. code-block:: yaml

    - decl: struct Cstruct1 {
              int ifield;
              double dfield;
            };

The ``struct`` statement can can also be used to declare a variable of
a previously defined structure.  This is required for C but is
optional for C++ where a ``struct`` statement defines a type.  To
distinguish a variable declaration from a struct declaration, the
trailing semicolon is required.

.. code-block:: yaml

    - decl: struct Cstruct1 var;


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

.. *wrap_class_as* also exists and defaults to *class*.

C
^

A C wrapper creates a ``struct`` with the same fields as the C++ struct.
The name of the ``struct`` is mangled so that any struct defined within
a namespace will be accessable as a global symbol for C.

Python
^^^^^^

Python can treat a struct in several different ways by setting option
*PY_struct_arg*.
First, treat it the same as a class.  An extension type is created with
descriptors for the field methods. Second, as a numpy descriptor.
This allows an array of structs to be used easily.
Finally, as a tuple of Python objects.

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

Member Variables
^^^^^^^^^^^^^^^^

Generally, getter and setter functions are not required since Fortran
can directly access the member fields.  But when the member is a
pointer it is more convient to have a getter and setter which works
with Fortran pointers.

.. code-block:: c++

    struct PointerData
    {
       int nitems;
       int *items;
    };

The generated getter and setter are not type-bound functions and must
be passed the struct variable:

.. code-block :: fortran

    type(PointerData) var
    integer(C_INT) :: nitems
    integer(C_INT) :: items(10)
    integer(C_INT), pointer :: out(:)

    var%nitems = 10
    call pointerdata_set_items(var, items)
    
    out => pointerdata_get_items(var)

The names of these functions are controlled by the options
*SH_struct_getter_template* and *SH_struct_setter_template*.
They are added to the same scope as the struct.
    
Option *F_struct_getter_setter* can be set to *false* to avoid
creating the getter and setter functions.

.. _struct_object_oriented_c:

Object-oriented C
-----------------

Object oriented programing is a model and not a language feature.
This model has been used for years in C by creating a struct for the
object, then functions for the methods. C++ will implicitly pass the
``this`` argument. C methods explicitly pass the struct as an
argument. Fortran and Python both pass an explicit object then wrap it
in syntacatic sugar to allow a ``self.method()`` syntax to be used.
Shroud allows a struct and collection of functions to be treated as a
class.

First, define a struct and set the *wrap_struct_as* options to *class*.

.. literalinclude:: ../regression/input/struct.yaml
   :language: yaml
   :start-after: start Cstruct_as_class
   :end-before: end Cstruct_as_class

Create a constructor function.
The *class_ctor* options associates this with the struct.

.. literalinclude:: ../regression/input/struct.yaml
   :language: yaml
   :start-after: start Create_Cstruct_as_class
   :end-before: end Create_Cstruct_as_class

Then add methods.  The *class_method* option associates this with the
struct.  The format field *F_name_function* is used to name the
method.  The default method name is the same as the function name.
But since this name will be used in the context of the object, it can
be much shorter.  The *pass* attribute marks this as the 'object'.

.. literalinclude:: ../regression/input/struct.yaml
   :language: yaml
   :start-after: start Cstruct_as_class_sum
   :end-before: end Cstruct_as_class_sum

Additonal options are *wrap_class_as* and *class_baseclass*.

Fortran
^^^^^^^

A *shadow* class is created for the struct.
This is the same as wrapping a C++ class.
Getters and setters are created for the member variable.
And the ``sum`` method is added.

.. literalinclude:: ../regression/reference/struct-c/wrapfstruct.f
   :language: fortran
   :start-after: start derived-type cstruct_as_class
   :end-before: end derived-type cstruct_as_class
   :dedent: 4

Now the struct is treated as a class in the Fortran wrapper.

.. code-block:: fortran

    use struct_mod
    type(cstruct_as_class) var     ! Create Fortran variable.
    integer(C_INT) i
    var = cstruct_as_class()       ! Create struct in C++.
    i = var%sum()


Similar to Python, Fortran passes the object as an explicit argument.
Unlike C++ which uses an implicit ``this`` variable.
By default, the first argument of the function is assumed to be the object.
However, this can be changed using the *pass* attribute. This will add the
Fortran keyword ``PASS`` to the corresponding argument.

A full example is at 
:ref:`Struct as a Class <example_struct_as_class>`.
    
