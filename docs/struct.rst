.. Copyright (c) 2017-2019, Lawrence Livermore National Security, LLC. 
..
.. Produced at the Lawrence Livermore National Laboratory 
..
.. LLNL-CODE-738041.
..
.. All rights reserved. 
..
.. This file is part of Shroud.
..
.. For details about use and distribution, please read LICENSE.
..
.. #######################################################################

.. _TypesSandC:

Structs and Classes
===================

Shroud supports both structs and classes. But it treats them much
differently.  Whereas in C++ a struct and class are essentially the
same thing, Shroud treats structs as a C style struct.  They do not
have associated methods.  This allows them to be mapped to a Fortran
derived type with the ``bind(C)`` attribute.  Classes are wrapped by a
shadow derived-type with methods implemented as type-bound procedures
in Fortran.

Struct
------



Class
-----

Each class in the input file will create a Fortran derived type which
acts as a shadow class for the C++ class.  A pointer to an instance is
saved as a ``type(C_PTR)`` value.

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

The Fortran interface will create two derived types.  The first is
used to interact with the C wrapper. The C wrapper creates a
corresponding struct.  It contains a pointer to an instance of the
class and index used to release the instance.

.. literalinclude:: ../regression/reference/tutorial/wrapftutorial.f
   :language: fortran
   :start-after: start derived-type SHROUD_class1_capsule
   :end-before: end derived-type SHROUD_class1_capsule
   :dedent: 4

.. literalinclude:: ../regression/reference/tutorial/typesTutorial.h
   :language: c
   :start-after: start struct TUT_class1
   :end-before: end struct TUT_class1


The capsule is added to the Fortran shadow class.  This derived type
can contain type-bound procedures and may not use the ``bind(C)``
attribute.

.. code-block:: fortran

    type class1
        type(SHROUD_class1_capsule) :: cxxmem
    contains
        procedure :: method1 => class1_method1
    end type class1


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
