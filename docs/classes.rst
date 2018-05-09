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

The method ``new`` has the attribute **+constructor** to mark it as a
constructor.  In this example the empty paren expression is required
to apply the annotation to the function instead of the result.
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
