.. Copyright (c) 2017-2020, Lawrence Livermore National Security, LLC and
   other Shroud Project Developers.
   See the top-level COPYRIGHT file for details.

   SPDX-License-Identifier: (BSD-3-Clause)

Tutorial
========

This tutorial will walk through the steps required to create a Fortran or
Python wrapper for a simple C++ library.

Functions
---------

The simplest item to wrap is a function in the file :file:`tutorial.hpp`:

.. code-block:: c++

    namespace tutorial {
      void NoReturnNoArguments(void);
    }

This is wrapped using a YAML input file :file:`tutorial.yaml`:

.. code-block:: yaml

    library: Tutorial
    cxx_header: tutorial.hpp

    declarations:
    - decl: namespace tutorial
      declarations:
      - decl: void NoReturnNoArguments()

.. XXX support (void)?

.. The **options** mapping allows the user to give information to guide the wrapping.

**library** is used to name output files and name the
Fortran module.  **cxx_header** is the name of a C++ header file which
contains the declarations for functions to be wrapped.  **declarations**
is a sequence of mappings which describe the functions to wrap.

Process the file with *Shroud*:

.. code-block:: sh

    % shroud tutorial.yaml
    Wrote wrapTutorial.h
    Wrote wrapTutorial.cpp
    Wrote wrapftutorial.f

    Wrote pyClass1type.cpp
    Wrote pyTutorialmodule.hpp
    Wrote pyTutorialmodule.cpp
    Wrote pyTutorialutil.cpp


The C++ code to call the function:

.. code-block:: c++

    #include "tutorial.hpp"

    using namespace tutorial;
    NoReturnNoArguments();

And the Fortran version:

.. code-block:: fortran

    use tutorial_mod
    call no_return_no_arguments

.. note :: rename module to just tutorial.

The generated code is listed at :ref:`NoReturnNoArguments <example_NoReturnNoArguments>`.

Arguments
---------

Integer and Real
^^^^^^^^^^^^^^^^

Integer and real types are handled using the ``iso_c_binding`` module
which match them directly to the corresponding types in C++.
To wrap ``PassByValue``:

.. code-block:: c++

    double PassByValue(double arg1, int arg2)
    {
        return arg1 + arg2;
    }

Add the declaration to the YAML file:

.. code-block:: yaml

    declarations:
    - decl: double PassByValue(double arg1, int arg2)

Usage:

.. code-block:: fortran

    use tutorial_mod
    real(C_DOUBLE) result
    result = pass_by_value(1.d0, 4)

.. code-block:: python

    import tutorial
    result = tutorial.PassByValue(1.0, 4)



Pointer Functions
-----------------

Functions which return a pointer will create a Fortran wrapper with
the ``POINTER`` attribute:

.. code-block:: yaml

    - decl: int * ReturnIntPtrDim(int *len+intent(out)+hidden) +dimension(len)

The C++ routine returns a pointer to an array and the length of the array
in argument ``len``.  The Fortran API does not need to pass the argument
since the returned pointer will know its length.
The *hidden* attribute will cause ``len`` to be omitted from the Fortran API,
but still passed to the C API.

It can be used as:

.. code-block:: fortran

    integer(C_INT), pointer :: intp(:)

    intp => return_int_ptr()


Pointer arguments
-----------------

When a C++ routine accepts a pointer argument it may mean
several things

 * output a scalar
 * input or output an array
 * pass-by-reference for a struct or class.

In this example, ``len`` and ``values`` are an input array and
``result`` is an output scalar:

.. code-block:: c++

    void Sum(size_t len, int *values, int *result)
    {
        int sum = 0;
        for (size_t i=0; i < len; i++) {
          sum += values[i];
        }
        *result = sum;
        return;
    }

When this function is wrapped it is necessary to give some annotations
in the YAML file to describe how the variables should be mapped to
Fortran:

.. code-block:: yaml

  - decl: void Sum(size_t len  +implied(size(values)),
                   int *values +dimension(:)+intent(in),
                   int *result +intent(out))

In the ``BIND(C)`` interface only *len* uses the ``value`` attribute.
Without the attribute Fortran defaults to pass-by-reference
i.e. passes a pointer.
The ``dimension`` attribute defines the variable as a one dimensional,
assumed-shape array.  In the C interface this maps to an 
assumed-length array.  C pointers, like assumed-length arrays, have no
idea how many values they point to.  This information is passed
by the *len* argument.

The *len* argument defines the ``implied`` attribute.  This argument
is not part of the Fortran API since its presence is *implied* from the
expression ``size(values)``. This uses the Fortran intrinsic ``size``
to compute the total number of elements in the array.  It then passes
this value to the C wrapper:

.. code-block:: fortran

    use tutorial_mod
    integer(C_INT) result
    call sum([1,2,3,4,5], result)

.. code-block:: python

    import tutorial
    result = tutorial.Sum([1, 2, 3, 4, 5])

See example :ref:`Sum <example_Sum>` for generated code.

String
^^^^^^

Character variables have significant differences between C and
Fortran.  The Fortran interoperability with C feature treats a
``character`` variable of default kind as an array of
``character(kind=C_CHAR,len=1)``.  The wrapper then deals with the C
convention of ``NULL`` termination to Fortran's blank filled.

C++ routine:

.. code-block:: c++

    const std::string ConcatenateStrings(
        const std::string& arg1,
        const std::string& arg2)
    {
        return arg1 + arg2;
    }

YAML input:

.. code-block:: yaml

    declarations:
    - decl: const std::string ConcatenateStrings(
        const std::string& arg1,
        const std::string& arg2 )

The function is called as:

.. code-block:: fortran

    character(len=:), allocatable :: rv4c

    rv4c = concatenate_strings("one", "two")

.. XXX fill in python example

.. note :: This function is just for demonstration purposes.
           Any reasonable person would just use the concatenation operator in Fortran.
 

Default Value Arguments
------------------------

Each function with default value arguments will create a C and Fortran 
wrapper for each possible prototype.  For Fortran, these functions
are then wrapped in a generic statement which allows them to be
called by the original name.
A header files contains:

.. code-block:: c++

    double UseDefaultArguments(double arg1 = 3.1415, bool arg2 = true)

and the function is defined as:

.. code-block:: c++

    double UseDefaultArguments(double arg1, bool arg2)
    {
        if (arg2) {
            return arg1 + 10.0;
        } else {
            return arg1;
        }
     }

Creating a wrapper for each possible way of calling the C++ function
allows C++ to provide the default values:

.. code-block:: yaml

    declarations:
    - decl: double UseDefaultArguments(double arg1 = 3.1415, bool arg2 = true)
      default_arg_suffix:
      -  
      -  _arg1
      -  _arg1_arg2

The *default_arg_suffix* provides a list of values of
*function_suffix* for each possible set of arguments for the function.
In this case 0, 1, or 2 arguments.

Fortran usage:

.. code-block:: fortran

  use tutorial_mod
  print *, use_default_arguments()
  print *, use_default_arguments(1.d0)
  print *, use_default_arguments(1.d0, .false.)

Python usage:

     >>> import tutorial
     >>> tutorial.UseDefaultArguments()
     13.1415
     >>> tutorial.UseDefaultArguments(1.0)
     11.0
     >>> tutorial.UseDefaultArguments(1.0, False)
     1.0

The generated code is listed at
:ref:`UseDefaultArguments <example_UseDefaultArguments>`.

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

C++:

.. code-block:: c++

    void OverloadedFunction(const std::string &name);
    void OverloadedFunction(int indx);

By default the names are mangled by adding an index to the end. This
can be controlled by setting **function_suffix** in the YAML file:

.. code-block:: yaml

  declarations:
  - decl: void OverloadedFunction(const std::string& name)
    function_suffix: _from_name
  - decl: void OverloadedFunction(int indx)
    function_suffix: _from_index

.. code-block:: fortran

  call overloaded_function_from_name("name")
  call overloaded_function_from_index(1)
  call overloaded_function("name")
  call overloaded_function(1)

.. code-block:: python

   tutorial.OverloadedFunction("name")
   tutorial.OverloadedFunction(1)

Optional arguments and overloaded functions
-------------------------------------------

Overloaded function that have optional arguments can also be wrapped:

.. code-block:: yaml

  - decl: int UseDefaultOverload(int num,
            int offset = 0, int stride = 1)
  - decl: int UseDefaultOverload(double type, int num,
            int offset = 0, int stride = 1)

These routines can then be called as:

.. code-block:: fortran

    rv = use_default_overload(10)
    rv = use_default_overload(1d0, 10)

    rv = use_default_overload(10, 11, 12)
    rv = use_default_overload(1d0, 10, 11, 12)

Templates
---------

C++ template are handled by creating a wrapper for each instantiation 
of the function defined by the **cxx_template** field.
The C and Fortran names are mangled by adding a type suffix to the function name.

C++:

.. code-block:: c++

  template<typename ArgType>
  void TemplateArgument(ArgType arg)
  {
      return;
  }

YAML:

.. code-block:: yaml

  - decl: |
        template<typename ArgType>
        void TemplateArgument(ArgType arg)
    cxx_template:
    - instantiation: <int>
    - instantiation: <double>

Fortran usage:

.. code-block:: fortran

    call template_argument(1)
    call template_argument(10.d0)

Python usage:

.. code-block:: python

        tutorial.TemplateArgument(1)
        tutorial.TemplateArgument(10.0)

Likewise, the return type can be templated but in this case no
interface block will be generated since generic function cannot vary
only by return type.

C++:

.. code-block:: c++

  template<typename RetType>
  RetType TemplateReturn()
  {
      return 0;
  }

YAML:

.. code-block:: yaml

  - decl: template<typename RetType> RetType TemplateReturn()
    cxx_template:
    - instantiation: <int>
    - instantiation: <double>

Fortran usage:

.. code-block:: fortran

    integer(C_INT) rv_integer
    real(C_DOUBLE) rv_double
    rv_integer = template_return_int()
    rv_double = template_return_double()

Python usage:

.. code-block:: python

    rv_integer = TemplateReturn_int()
    rv_double = TemplateReturn_double()

Generic Functions
-----------------

C and C++ provide a type promotion feature when calling functions
which Fortran does not support:

.. code-block:: fortran

    void FortranGeneric(double arg);

    FortranGeneric(1.0f);
    FortranGeneric(2.0);

When ``FortranGeneric`` is wrapped in Fortran it may only be used with
the correct arguments:

.. code-block:: sh

    call fortran_generic(1.)
                         1
    Error: Type mismatch in argument 'arg' at (1); passed REAL(4) to REAL(8)

It would be possible to create a version of the routine in C++ which
accepts floats, but that would require changes to the library being
wrapped.  Instead it is possible to create a generic interface to the
routine by defining which variables need their types changed.  This is
similar to templates in C++ but will only impact the Fortran wrapper.
Instead of specify the Type which changes, you specify the argument which changes:

.. code-block:: yaml

  - decl: void FortranGeneric(double arg)
    fortran_generic:
    - decl: (float arg)
      function_suffix: float
    - decl: (double arg)
      function_suffix: double

It may now be used with single or double precision arguments:

.. code-block:: fortran

  call fortran_generic(1.0)
  call fortran_generic(1.0d0)

A full example is at :ref:`GenericReal <example_GenericReal>`.

Types
-----


Typedef
^^^^^^^

Sometimes a library will use a ``typedef`` to identify a specific
use of a type:

.. code-block:: c++

    typedef int TypeID;

    int typefunc(TypeID arg);

Shroud must be told about user defined types in the YAML file:

.. code-block:: yaml

    declarations:
    - decl: typedef int TypeID;

This will map the C++ type ``TypeID`` to the predefined type ``int``.
The C wrapper will use ``int``:

.. code-block:: c++

    int TUT_typefunc(int arg)
    {
        tutorial::TypeID SHC_rv = tutorial::typefunc(arg);
        return SHC_rv;
    }

Enumerations
^^^^^^^^^^^^

Enumeration types can also be supported by describing the type to
shroud.
For example:

.. code-block:: c++

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
describe how to convert between C and C++:

.. code-block:: yaml

    declarations:
    - decl: typedef int EnumTypeID
      fields:
        c_to_cxx : static_cast<tutorial::EnumTypeID>({c_var})
        cxx_to_c : static_cast<int>({cxx_var})

The typename must be fully qualified
(use ``tutorial::EnumTypeId`` instead of ``EnumTypeId``).
The C argument is explicitly converted to a C++ type, then the
return type is explicitly converted to a C type in the generated wrapper:

.. code-block:: c++

  int TUT_enumfunc(int arg)
  {
      tutorial::EnumTypeID SHCXX_arg = static_cast<tutorial::EnumTypeID>(arg);
      tutorial::EnumTypeID SHCXX_rv = tutorial::enumfunc(SHCXX_arg);
      int SHC_rv = static_cast<int>(SHCXX_rv);
      return SHC_rv;
  }

Without the explicit conversion you're likely to get an error such as::

    error: invalid conversion from ‘int’ to ‘tutorial::EnumTypeID’

A enum can also be fully defined to Fortran:

.. code-block:: yaml

    declarations:
    - decl: |
          enum Color {
            RED,
            BLUE,
            WHITE
          };

In this case the type is implicitly defined so there is no need to add
it to the *types* list.  The C header duplicates the enumeration, but
within an ``extern "C"`` block:

.. code-block:: c++

    //  tutorial::Color
    enum TUT_Color {
        TUT_tutorial_Color_RED,
        TUT_tutorial_Color_BLUE,
        TUT_tutorial_Color_WHITE
    };

Fortran creates integer parameters for each value:

.. code-block:: fortran

    !  enum tutorial::Color
    integer(C_INT), parameter :: tutorial_color_red = 0
    integer(C_INT), parameter :: tutorial_color_blue = 1
    integer(C_INT), parameter :: tutorial_color_white = 2


.. note:: Fortran's ``ENUM, BIND(C)`` provides a way of matching 
          the size and values of enumerations.  However, it doesn't
          seem to buy you too much in this case.  Defining enumeration
          values as ``INTEGER, PARAMETER`` seems more straightforward.

Structure
^^^^^^^^^

A structure in C++ can be mapped directly to a Fortran derived type using the 
``bind(C)`` attribute provided by Fortran 2003. For example, the C++ code:

.. code-block:: c++

    struct struct1 {
      int ifield;
      double dfield;
    };

can be defined to Shroud with the YAML input:

.. code-block:: yaml

    - decl: |
        struct struct1 {
          int ifield;
          double dfield;
        };

This will generate a C struct which is compatible with C++:

.. code-block:: c++

    struct s_TUT_struct1 {
        int ifield;
        double dfield;
    };
    typedef struct s_TUT_struct1 TUT_struct1;

A C++ struct is compatible with C; however, its name may not be accessible to
C since it may be defined within a namespace.  By creating an identical struct in the 
C wrapper, we're guaranteed visibility for the C API.

.. note:: All fields must be defined in the YAML file in order to ensure that
          ``sizeof`` operator will return the same value for the C and C++ structs.

This will generate a Fortran derived type which is compatible with C++:

.. code-block:: fortran

    type, bind(C) :: struct1
        integer(C_INT) :: ifield
        real(C_DOUBLE) :: dfield
    end type struct1

A function which returns a struct value can have its value copied into a
Fortran variable where the fields can be accessed directly by Fortran.
A C++ function which initialized a struct can be written as:

.. code-block:: yaml

    - decl: struct1 returnStructByValue(int i, double d);

The C wrapper casts the C++ struct to the C struct by using
pointers to the struct then returns the value by dereferencing
the C struct pointer.

.. code-block:: c++

    TUT_struct1 TUT_return_struct_by_value(int i, double d)
    {
        Cstruct1 SHCXX_rv = returnStructByValue(i, d);
        TUT_cstruct1 * SHC_rv = static_cast<TUT_cstruct1 *>(
            static_cast<void *>(&SHCXX_rv));
        return *SHC_rv;
    }

This function can be called directly by Fortran using the generated
interface:

.. code-block:: fortran

        function return_struct_by_value(i, d) &
                result(SHT_rv) &
                bind(C, name="TUT_return_struct_by_value")
            use iso_c_binding, only : C_DOUBLE, C_INT
            import :: struct1
            implicit none
            integer(C_INT), value, intent(IN) :: i
            real(C_DOUBLE), value, intent(IN) :: d
            type(struct1) :: SHT_rv
        end function return_struct

To use the function:

.. code-block:: fortran

    type(struct1) var

    var = return_struct(1, 2.5)
    print *, var%ifield, var%dfield


Classes
-------

Each class is wrapped in a Fortran derived type which shadows the C++
class by holding a ``type(C_PTR)`` pointer to an C++ instance.  Class
methods are wrapped using Fortran's type-bound procedures.  This makes
Fortran usage very similar to C++.

Now we'll add a simple class to the library:

.. code-block:: c++

    class Class1
    {
    public:
        void Method1() {};
    };

To wrap the class add the lines to the YAML file:

.. code-block:: yaml

    declarations:
    - decl: class Class1
      declarations:
      - decl: Class1()  +name(new)
        format:
          function_suffix: _default
      - decl: ~Class1() +name(delete)
      - decl: int Method1()

The constructor and destructor have no method name associated with
them.  They default to **ctor** and **dtor**.  The names can be
overridden by supplying the **+name** annotation.  These declarations
will create wrappers over the ``new`` and ``delete`` C++ keywords.

The C++ code to call the function:

.. code-block:: c++

    #include <tutorial.hpp>
    tutorial::Class1 *cptr = new tutorial::Class1();

    cptr->Method1();

And the Fortran version:

.. code-block:: fortran

    use tutorial_mod
    type(class1) cptr

    cptr = class1_new()
    call cptr%method1

Python usage:

.. code-block:: python

    import tutorial
    obj = tutorial.Class1()
    obj.method1()


Class static methods
^^^^^^^^^^^^^^^^^^^^

Class static methods are supported using the ``NOPASS`` keyword in Fortran.
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

Called from Fortran as:

.. code-block:: fortran

    type(singleton) obj0
    obj0 = obj0%get_reference()

Note that obj0 is not assigned a value before the function ``get_reference`` is called.
