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

Fortran Tutorial
================

This tutorial will walk through the steps required to create a Fortran
wrapper for a simple C++ library.

Functions
---------

The simplest item to wrap is a function in the file ``tutorial.hpp``::

   namespace tutorial {
     void Function1(void);
   }

This is wrapped using a YAML input file ``tutorial.yaml``::

  library: Tutorial
  cxx_header: tutorial.hpp
  namespace: tutorial

  functions:
  - decl: void Function1()

.. XXX support (void)?

.. The **options** mapping allows the user to give information to guide the wrapping.

**library** is used to name output files and name the
Fortran module.  **cxx_header** is the name of a C++ header file which
contains the declarations for functions to be wrapped.  **functions**
is a sequence of mappings which describe the functions to wrap.

Process the file with *Shroud*::

    % shroud tutorial.yaml
    Wrote wrapTutorial.h
    Wrote wrapTutorial.cpp
    Wrote wrapftutorial.f

The generated C function in file ``wrapTutorial.cpp`` is::

    #include "wrapTutorial.h"
    #include "tutorial.hpp"

    extern "C" {
    namespace tutorial {

    void TUT_function1()
    {
        Function1();
        return;
    }

    }  // namespace tutorial
    }  // extern "C"

To help control the scope of C names, all externals add a prefix.
It defaults to the first three letters of the
**library** but may be changed by setting the format **C_prefix**::

    format:
      C_prefix: NEW_

The Fortran module in ``wrapftutorial.f`` contains an interface
which allows the C wrapper to be called directly by Fortran::

    module tutorial_mod
        implicit none

        interface
            subroutine function1() &
                    bind(C, name="TUT_function1")
                use iso_c_binding
                implicit none
            end subroutine function1
        end interface
    contains
    end module tutorial_mod

In other cases a Fortran wrapper will also be created which will 
do some type conversion on arguments or results 
before or after calling the C wrapper.  It may also be used
to pass additional information to the C wrapper such as a ``CHARACTER``
variable ``LEN`` or an array's ``SIZE``.

The C++ code to call the function::

    #include "tutorial.hpp"

    using namespace tutorial;
    Function1();

And the Fortran version::

    use tutorial_mod

    call function1

.. note :: rename module to just tutorial.


Arguments
---------

Integer and Real
^^^^^^^^^^^^^^^^

Integer and real types are handled using the ``iso_c_binding`` module
which match them directly to the corresponding types in C++.
To wrap ``Function2``::

    double Function2(double arg1, int arg2)
    {
        return arg1 + arg2;
    }

Add the declaration to the YAML file::

    functions:
    - decl: double Function2(double arg1, int arg2)

The arguments are added to the interface for the C routine using the
``value`` attribute.  They use the ``intent(IN)`` attribute since they
are pass-by-value and cannot return a value.
The C wrapper can be called directly by Fortran using the interface::

     interface
        function function2(arg1, arg2) &
                result(SHT_rv) &
                bind(C, name="TUT_function2")
            use iso_c_binding, only : C_DOUBLE, C_INT
            implicit none
            real(C_DOUBLE), value, intent(IN) :: arg1
            integer(C_INT), value, intent(IN) :: arg2
            real(C_DOUBLE) :: SHT_rv
        end function function2
     end interface


Pointer arguments
-----------------

When a C++ routine accepts a pointer argument it may mean
several things

 * output a scalar
 * input or output an array
 * pass-by-reference for a struct or class.

In this example, ``len`` and ``values`` are an input array and
``result`` is an output scalar::

    void Sum(int len, int *values, int *result)
    {
        int sum = 0;
        for (int i=0; i < len; i++) {
          sum += values[i];
        }
        *result = sum;
        return;
    }

When this function is wrapped it is necessary to give some annotations
in the YAML file to describe how the variables should be mapped to
Fortran::

  - decl: void Sum(int len, int *values+dimension(len)+intent(in),
                   int *result+intent(out))

In the ``BIND(C)`` interface only *len* uses the ``value`` attribute.
Without the attribute Fortran defaults to pass-by-reference i.e.
passes a pointer::

    interface
        subroutine sum(len, values, result) &
                bind(C, name="TUT_sum")
            use iso_c_binding
            implicit none
            integer(C_INT), value, intent(IN) :: len
            integer(C_INT), intent(IN) :: values(len)
            integer(C_INT), intent(OUT) :: result
        end subroutine sum
    end interface

.. note:: Multiply pointered arguments ( ``char **`` ) do not 
          map to Fortran directly and require ``type(C_PTR)``.

Logical
^^^^^^^

Logical variables require a conversion since they are not directly
compatible with C.  In addition, how ``.true.`` and ``.false.`` are
represented internally is compiler dependent.  Some compilers use 1 for
``.true.`` while other use -1.

A simple C++ function which accepts and returns a boolean argument::

    bool Function3(bool arg)
    {
        return ! arg;
    }

Added to the YAML file as before::

    functions:
    - decl: bool Function3(bool arg)


In this case a Fortran wrapper is created in addition to the interface.
The wrapper convert the logical's value before calling the C wrapper::

     interface
        function c_function3(arg) &
                result(SHT_rv) &
                bind(C, name="TUT_function3")
            use iso_c_binding, only : C_BOOL
            implicit none
            logical(C_BOOL), value, intent(IN) :: arg
            logical(C_BOOL) :: SHT_rv
        end function c_function3
    end interface

    function function3(arg) &
            result(SHT_rv)
        use iso_c_binding, only : C_BOOL
        logical, value, intent(IN) :: arg
        logical(C_BOOL) SH_arg
        logical :: SHT_rv
        SH_arg = arg  ! coerce to C_BOOL
        SHT_rv = c_function3(SH_arg)
    end function function3

The wrapper routine uses the compiler to coerce type using an assignment.
It is possible to call ``c_function3`` directly from Fortran, but the
wrapper does the type conversion necessary to make it easier to use
within an existing Fortran application.


Character
^^^^^^^^^

Character variables have significant differences between C and
Fortran.  The Fortran interoperability with C feature treats a
``character`` variable of default kind as an array of
``character(kind=C_CHAR,len=1)``.  The wrapper then deals with the C
convention of ``NULL`` termination to Fortran's blank filled.

C++ routine::

    const std::string Function4a(
        const std::string& arg1,
        const std::string& arg2)
    {
        return arg1 + arg2;
    }

YAML input::

    functions
    - decl: const std::string Function4a+len(30)(
        const std::string& arg1,
        const std::string& arg2 )

This is the C++ prototype with the addition of **+len(30)**.  This
attribute defines the declared length of the returned string.  Since
*Function4a* is returning a ``std::string`` the contents of the string
must be copied out into a Fortran variable so that the ``std::string``
may be deallocated by C++. Otherwise, it would leak memory.

Attributes may also be added by assign new fields in **attrs**::

    - decl: const std::string Function4a(
        const std::string& arg1,
        const std::string& arg2 )
      attrs:
        result:
          len: 30

The C wrapper uses ``char *`` for ``std::string`` arguments which
Fortran declares as ``character``.
The argument is passed to the ``std::string`` constructor.
In addition the length of the data in each string is computed using ``len_trim``
and passed down.
No trailing ``NULL`` is required.
This avoids copying the string in Fortran which would be necessary to
append the trailing ``C_NULL_CHAR``.
The return value is added as another argument along with its declared length
computed using ``len``::

    void TUT_function4a_bufferify(
        const char * arg1, int Larg1,
        const char * arg2, int Larg2,
        char * SHF_rv, int NSHF_rv)
    {
        const std::string SH_arg1(arg1, Larg1);
        const std::string SH_arg2(arg2, Larg2);
        const std::string SHT_rv = Function4a(SH_arg1, SH_arg2);
        if (SHT_rv.empty()) {
            std::memset(SHF_rv, ' ', NSHF_rv);
        } else {
            ShroudStrCopy(SHF_rv, NSHF_rv, SHT_rv.c_str());
        }
        return;
    }

The contents of the ``std::string`` are copied into the result argument and blank
filled by ``ShroudStrCopy``.
Before the C wrapper returns, ``SHT_rv`` will be deleted.

The Fortran wrapper::

    function function4a(arg1, arg2) &
            result(SHT_rv)
        use iso_c_binding, only : C_CHAR, C_INT
        character(*), intent(IN) :: arg1
        character(*), intent(IN) :: arg2
        character(kind=C_CHAR, len=30) :: rv
        call c_function4a_bufferify(arg1, len_trim(arg1, kind=C_INT),  &
            arg2, len_trim(arg2, kind=C_INT), SHT_rv, &
            len(SHT_rv, kind=C_INT)))
    end function function4a

The function is called as::

  character(30) rv4a

  rv4a = function4a("bird", "dog")

.. note :: This function is just for demonstration purposes.
           Any reasonable person would just use the concatenation operator in Fortran.

Default Value Arguments
------------------------

Each function with default value arguments will create a C and Fortran 
wrapper for each possible prototype.  For Fortran, these functions
are then wrapped in a generic statement which allows them to be
called by the original name.
Creating a wrapper for each possible way of calling the C++ function
allows C++ to provide the default values::

    functions:
    - decl: double Function5(double arg1 = 3.1415, bool arg2 = true)
      default_arg_suffix:
      -  
      -  _arg1
      -  _arg1_arg2

The *default_arg_suffix* provides a list of values of
*function_suffix* for each possible set of arguments for the function.
In this case 0, 1, or 2 arguments.

C wrappers::

    double TUT_function5()
    {
        double SHT_rv = Function5();
        return SHT_rv;
    }
    
    double TUT_function5_arg1(double arg1)
    {
        double SHT_rv = Function5(arg1);
        return SHT_rv;
    }
    
    double TUT_function5_arg1_arg2(double arg1, bool arg2)
    {
        double SHT_rv = Function5(arg1, arg2);
        return SHT_rv;
    }


Fortran wrapper::

    interface function5
        module procedure function5
        module procedure function5_arg1
        module procedure function5_arg1_arg2
    end interface function5

    contains

    function function5() &
            result(SHT_rv)
        use iso_c_binding, only : C_DOUBLE
        real(C_DOUBLE) :: SHT_rv
        SHT_rv = c_function5()
    end function function5
    
    function function5_arg1(arg1) &
            result(SHT_rv)
        use iso_c_binding, only : C_DOUBLE
        real(C_DOUBLE), value, intent(IN) :: arg1
        real(C_DOUBLE) :: SHT_rv
        SHT_rv = c_function5_arg1(arg1)
    end function function5_arg1
    
    function function5_arg1_arg2(arg1, arg2) &
            result(SHT_rv)
        use iso_c_binding, only : C_BOOL, C_DOUBLE
        real(C_DOUBLE), value, intent(IN) :: arg1
        logical, value, intent(IN) :: arg2
        logical(C_BOOL) SH_arg2
        real(C_DOUBLE) :: SHT_rv
        SH_arg2 = arg2  ! coerce to C_BOOL
        SHT_rv = c_function5_arg1_arg2(arg1, tmp_arg2)
    end function function5_arg1_arg2

Fortran usage::

  print *, function5()
  print *, function5(1.d0)
  print *, function5(1.d0, .false.)

.. note :: Fortran's ``OPTIONAL`` attribute provides similar but
           different semantics.
           Creating wrappers for each set of arguments allows
           C++ to supply the default value.  This is important
           when the default value does not map directly to Fortran.
           For example, ``bool`` type or when the default value
           is created by calling a C++ function.

           Using the ``OPTIONAL`` keyword creates the possibility to
           call the C++ function in a way which is not supported by
           the C++ compilers.
           For example, ``function5(arg2=.false.)``

           Fortran has nothing similar to variadic functions.

Overloaded Functions
--------------------

C++ allows function names to be overloaded.  Fortran supports this
by using a ``generic`` interface.  The C and Fortran wrappers will
generated a wrapper for each C++ function but must mangle the name to
distinguish the names.

C++::

    void Function6(const std::string &name);
    void Function6(int indx);

By default the names are mangled by adding an index to the end. This
can be controlled by setting **function_suffix** in the YAML file::

  functions:
  - decl: void Function6(const std::string& name)
    function_suffix: _from_name
  - decl: void Function6(int indx)
    function_suffix: _from_index

The generated C wrappers uses the mangled name::

    void TUT_function6_from_name(const char * name)
    {
        const std::string SH_name(name);
        Function6(SH_name);
        return;
    }

    void TUT_function6_from_index(int indx)
    {
        Function6(indx);
        return;
    }

The generated Fortran creates routines with the same mangled names but
also creates a generic interface block to allow them to be called by
the overloaded name::

    interface function6
        module procedure function6_from_name
        module procedure function6_from_index
    end interface function6

They can be used as::

  call function6_from_name("name")
  call function6_from_index(1)
  call function6("name")
  call function6(1)

Optional arguments and overloaded functions
-------------------------------------------

Overloaded function that have optional arguments can also be wrapped::

  - decl: int overload1(int num,
            int offset = 0, int stride = 1)
  - decl: int overload1(double type, int num,
            int offset = 0, int stride = 1)

These routines can then be called as::

    rv = overload1(10)
    rv = overload1(1d0, 10)

    rv = overload1(10, 11, 12)
    rv = overload1(1d0, 10, 11, 12)

Templates
---------

C++ template are handled by creating a wrapper for each instantiation 
of the function defined by the **cxx_template** field.
The C and Fortran names are mangled by adding a type suffix to the function name.

C++::

  template<typename ArgType>
  void Function7(ArgType arg)
  {
      return;
  }

YAML::

  - decl: void Function7(ArgType arg)
    cxx_template:
      ArgType:
        - int
        - double

C wrapper::

    void TUT_function7_int(int arg)
    {
        Function7<int>(arg);
        return;
    }
    
    void TUT_function7_double(double arg)
    {
        Function7<double>(arg);
        return;
    }

The Fortran wrapper will also generate an interface block::

    interface function7
        module procedure function7_int
        module procedure function7_double
    end interface function7


Likewise, the return type can be templated but in this case no
interface block will be generated since generic function cannot vary
only by return type.


C++::

  template<typename RetType>
  RetType Function8()
  {
      return 0;
  }

YAML::

  - decl: RetType Function8()
    cxx_template:
      RetType:
        - int
        - double

C wrapper::

    int TUT_function8_int()
    {
        int SHT_rv = Function8<int>();
        return SHT_rv;
    }

    double TUT_function8_double()
    {
        double SHT_rv = Function8<double>();
        return SHT_rv;
    }

Generic Functions
-----------------

C and C++ provide a type promotion feature when calling functions
which Fortran does not support::

    void Function9(double arg);

    Function9(1.0f);
    Function9(2.0);

When Function9 is wrapped in Fortran it may only be used with the correct arguments::

    call function9(1.)
                   1
  Error: Type mismatch in argument 'arg' at (1); passed REAL(4) to REAL(8)

It would be possible to create a version of the routine in C++ which
accepts floats, but that would require changes to the library being
wrapped.  Instead it is possible to create a generic interface to the
routine by defining which variables need their types changed.  This is
similar to templates in C++ but will only impact the Fortran wrapper.
Instead of specify the Type which changes, you specify the argument which changes::

  - decl: void Function9(double arg)
    fortran_generic:
       arg:
       -  float
       -  double

This will generate only one C wrapper which accepts a double::

  void TUT_function9(double arg)
  {
      Function9(arg);
      return;
  }

But it will generate two Fortran wrappers and a generic interface
block.  Each wrapper will coerce the argument to the correct type::

    interface function9
        module procedure function9_float
        module procedure function9_double
    end interface function9

    subroutine function9_float(arg)
        use iso_c_binding, only : C_DOUBLE, C_FLOAT
        real(C_FLOAT), value, intent(IN) :: arg
        call c_function9(real(arg, C_DOUBLE))
    end subroutine function9_float
    
    subroutine function9_double(arg)
        use iso_c_binding, only : C_DOUBLE
        real(C_DOUBLE), value, intent(IN) :: arg
        call c_function9(arg)
    end subroutine function9_double

It may now be used with single or double precision arguments::

  call function9(1.0)
  call function9(1.0d0)


Types
-----


Typedef
^^^^^^^

Sometimes a library will use a ``typedef`` to identify a specific
use of a type::

    typedef int TypeID;

    int typefunc(TypeID arg);

Shroud must be told about user defined types in the YAML file::

  types:
    TypeID:
      typedef  : int
      cxx_type : TypeID

This will map the C++ type ``TypeID`` to the predefined type ``int``.
The C wrapper will use ``int``::

  int TUT_typefunc(int arg)
  {
    int SHT_rv = typefunc(arg);
    return SHT_rv;
  }

Enumerations
^^^^^^^^^^^^

Enumeration types can also be supported by describing the type to
shroud.
For example::

  namespace tutorial
  {

  enum EnumTypeID {
      ENUM0,
      ENUM1,
      ENUM2
  };

  EnumTypeID enumfunc(EnumTypeID arg);

  } /* end namespace tutorial */

This enumeration is within a namespace so it is not available to
C.  For C and Fortran the type can be describe as an ``int``
similar to how the ``typedef`` is defined. But in addition we
describe how to convert between C and C++::

    types:
      EnumTypeID:
        typedef  : int
        cxx_type : EnumTypeID
        c_to_cxx : static_cast<EnumTypeID>({c_var})
        cxx_to_c : static_cast<int>({cxx_var})

The C argument is explicitly converted to a C++ type, then the
return type is explicitly converted to a C type in the generated wrapper::

  int TUT_enumfunc(int arg)
  {
    EnumTypeID SHT_rv = enumfunc(static_cast<EnumTypeID>(arg));
    int XSHT_rv = static_cast<int>(SHT_rv);
    return XSHT_rv;
  }

Without the explicit conversion you're likely to get an error such as::

  error: invalid conversion from ‘int’ to ‘tutorial::EnumTypeID’

.. note:: Currently only the ``typedef`` is supported. There is no support
          for adding the enumeration values for C and Fortran.

          Fortran's ``ENUM, BIND(C)`` provides a way of matching 
          the size and values of enumerations.  However, it doesn't
          seem to buy you too much in this case.  Defining enumeration
          values as ``INTEGER, PARAMETER`` seems more straightforward.

Structure
^^^^^^^^^

TODO

Classes
-------

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

    classes:
    - name: Class1
      functions:
      - decl: Class1()  +name(new)
      - decl: ~Class1() +name(delete)
      - decl: void Method1()

The constructor and destructor have no method name associated with
them.  They default to **ctor** and **dtor**.  The names can be
overridden by supplying the **+name** annotation.  These declarations
will create wrappers over the ``new`` and ``delete`` C++ keywords.

The file ``wrapClass1.h`` will have an opaque struct for the class.
This is to allows some measure of type safety over using ``void``
pointers for every instance::

    struct s_TUT_class1;
    typedef struct s_TUT_class1 TUT_class1;


    TUT_class1 * TUT_class1_new()
    {
        Class1 *SHT_rv = new Class1();
        return static_cast<TUT_class1 *>(static_cast<void *>(SHT_rv));
    }

    void TUT_class1_delete(TUT_class1 * self)
    {
        Class1 *SH_this = static_cast<Class1 *>(static_cast<void *>(self));
        delete SH_this;
        return;
    }

    void TUT_class1_method1(TUT_class1 * self)
    {
        Class1 *SH_this = static_cast<Class1 *>(static_cast<void *>(self));
        SH_this->Method1();
        return;
    }

For Fortran a derived type is created::

    type class1
        type(C_PTR) voidptr
    contains
        procedure :: method1 => class1_method1
    end type class1

And the subroutines::

    function class1_new() &
            result(SHT_rv)
        type(class1) :: SHT_rv
        SHT_rv%voidptr = c_class1_new()
    end function class1_new
    
    subroutine class1_delete(obj)
        use iso_c_binding, only : C_NULL_PTR
        class(class1) :: obj
        call c_class1_delete(obj%voidptr)
        obj%voidptr = C_NULL_PTR
    end subroutine class1_delete

    subroutine class1_method1(obj)
        class(class1) :: obj
        call c_class1_method1(obj%voidptr)
    end subroutine class1_method1


The C++ code to call the function::

    tutorial::Class1 *cptr = new tutorial::Class1();

    cptr->Method1();

And the Fortran version::

    type(class1) cptr

    cptr = class1_new()
    call cptr%method1

