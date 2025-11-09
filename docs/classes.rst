.. Copyright Shroud Project Developers. See LICENSE file for details.

   SPDX-License-Identifier: (BSD-3-Clause)

:orphan:

.. from tutorial.rst  not sure if it deserves its own page

Classes
=======

Each class is wrapped in a Fortran derived type which holds a
``type(C_PTR)`` pointer to an C++ instance of the class.  Class
methods are wrapped using Fortran's type-bound procedures.  This makes
Fortran usage very similar to C++.

Now we'll add a simple class to the library::

    class Class1
    {
    public:
        void Method1() {};
    };

To wrap the class add the lines to the YAML file::

    declarations:
    - class: Class1
      declarations:
      - decl: Class1 new()  +name(new)
        format:
          function_suffix: _default
      - decl: ~Class1()  +name(delete)
      - decl: void Method1()

Shroud adds the attribute **+intent(ctor)** to the method ``new`` to mark it as a
constructor.
Likewise, ``delete`` is marked as a destructor.  These annotations
will create wrappers over the ``new`` and ``delete`` keywords.

The file ``wrapClass1.h`` will have an opaque struct for the class.
This is to allows some measure of type safety over using ``void``
pointers for every instance::

    struct s_TUT_class1 {
        void *addr;  /* address of C++ memory */
        int idtor;   /* index of destructor */
    };
    typedef struct s_TUT_class1 TUT_class1;


    TUT_class1 TUT_class1_new_default()
    {
        tutorial::Class1 *SHCXX_rv = new tutorial::Class1();
        TUT_class1 SHC_rv = { static_cast<void *>(SHCXX_rv), 0 };
        return SHC_rv;
    }

    void TUT_class1_method1(TUT_class1 * self)
    {
        tutorial::Class1 *SH_this = static_cast<tutorial::Class1 *>(self->addr);
        int SHC_rv = SH_this->Method1();
        return SHC_rv;
    }

For Fortran a derived type is created::

    type class1
        type(SHROUD_capsule_data), private :: cxxmem
    contains
        procedure :: method1 => class1_method1
    end type class1

And the subroutines::

    function class1_new_default() &
            result(SHT_rv)
        type(class1) :: SHT_rv
        SHT_rv%cxxmem = c_class1_new_default()
    end function class1_new
    
    function class1_method1(obj) &
            result(SHT_rv)
        use iso_c_binding, only : C_INT
        class(class1) :: obj
        integer(C_INT) :: SHT_rv
        SHT_rv = c_class1_method1(obj%cxxmem)
    end function class1_method1

The additional C++ code to call the function::

    tutorial::Class1 *cptr = new tutorial::Class1();

    cptr->Method1();

And the Fortran version::

    type(class1) cptr

    cptr = class1_new()
    call cptr%method1


.. Overload operators

   Adding the +operator(assignment) attribute will add
   generic :: assignment(=) => assign_weak.
   The function is assumed to have the correct arguments.
