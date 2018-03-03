.. Copyright (c) 2018, Lawrence Livermore National Security, LLC. 
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

Python Tutorial
===============

This tutorial will walk through the steps required to create a Python
wrapper for a simple C++ library.
The same input file is used as the Fortran tutorial.
This tutorial shows the generated wrapper code.
Some users may not care for this level of detail.
The intent of Shroud is to create the code that the user would be
required to write if they used the Python C API directly.

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

  options:
    wrap_fortran: False
    wrap_c: False
    wrap_python: True
    debug: True

  functions:
  - decl: void Function1()

.. XXX support (void)?

.. The **options** mapping allows the user to give information to guide the wrapping.

**library** is used to name output files and name the
Fortran module.  **cxx_header** is the name of a C++ header file which
contains the declarations for functions to be wrapped.  **functions**
is a sequence of mappings which describe the functions to wrap.

The default behavior of Shroud is to create a Fortran wrapper.  These options
are set to False and **wrap_python** is set to True.  In addition, the **debug**
option inserts some additional comments into the code that make it clearer 
where blocks of code are inserted.

Process the file with *Shroud*::

    % shroud tutorial.yaml
    Wrote pyClass1type.cpp
    Wrote pyTutorialmodule.hpp
    Wrote pyTutorialmodule.cpp
    Wrote pyTutorialhelper.cpp

The generated C function in file ``pyTutorialmodule.cpp`` is::

    static PyObject *
    PY_Function1(
      PyObject *SHROUD_UNUSED(self),
      PyObject *SHROUD_UNUSED(args),
      PyObject *SHROUD_UNUSED(kwds))
    {
        Function1();
        Py_RETURN_NONE;
    }

The wrapper implementation function is named using the format **PY_name_impl**
which defaults to ``{PY_prefix}{class_prefix}{function_name}{function_suffix}``.
This can be changed on a global level by resetting the template option::

    options:
      PY_name_impl_template: {PY_prefix}{class_prefix}{function_name}{function_suffix}_extra

All of the functions in the wrapper are file static exception for the module 
initialization function. To give them a unique name from the function which
is being wrapped the prefix **PY_format** is used.  It defaults to ``PY_``
but can be set to another value::

    format:
      PY_prefix: NEW_

Since the prototype of the function is required by the Python API,
the ``SHROUD_UNUSED`` macro is used to help avoid some compiler errors
about unused arguments.

The function uses the macro ``Py_RETURN_NONE`` from the Python API
to indicate a successful executation that returns no values.

Some additional boiler plate is created for the function::

    static PyMethodDef PY_methods[] = {
        {"Function1", (PyCFunction)PY_Function1, METH_NOARGS,
            PY_Function1__doc__},
        {NULL,   (PyCFunction)NULL, 0, NULL}            /* sentinel */
    };

Finally the module creation function is added at the end of the file::

    PyMODINIT_FUNC
    #ifdef IS_PY3K
    PyInit_tutorial(void)
    #else
    inittutorial(void)
    #endif
    {
        PyObject *m = NULL;
        const char * error_name = "tutorial.Error";
    
        /* Create the module and add the functions */
    #ifdef IS_PY3K
        m = PyModule_Create(&moduledef);
    #else
        m = Py_InitModule4("tutorial", PY_methods,
                           PY__doc__,
                           (PyObject*)NULL,PYTHON_API_VERSION);
    #endif
        if (m == NULL)
            return RETVAL;
        struct module_state *st = GETSTATE(m);
    
        PY_error_obj = PyErr_NewException((char *) error_name, NULL, NULL);
        if (PY_error_obj == NULL)
            return RETVAL;
        st->error = PY_error_obj;
        PyModule_AddObject(m, "Error", st->error);

        /* Check for errors */
        if (PyErr_Occurred())
            Py_FatalError("can't initialize module tutorial");
        return RETVAL;
    }


Arguments
---------


Integer and Real
^^^^^^^^^^^^^^^^

Arguments are parsed using ``PyArg_ParseTupleAndKeywords``
To wrap ``Function2``::

    double Function2(double arg1, int arg2)
    {
        return arg1 + arg2;
    }

Add the declaration to the YAML file::

    functions:
    - decl: double Function2(double arg1, int arg2)

Local variables are created for the argument values.
There values are filled in by ``PyArg_ParseTupleAndKeywords``.
The generated function is::

    static PyObject *
    PY_Function2(
      PyObject *SHROUD_UNUSED(self),
      PyObject *args,
      PyObject *kwds)
    {
        double arg1;
        int arg2;
        const char *SHT_kwlist[] = {
            "arg1",
            "arg2",
            NULL };
    
        if (!PyArg_ParseTupleAndKeywords(args, kwds, "di:Function2",
            const_cast<char **>(SHT_kwlist), &arg1, &arg2))
            return NULL;
    
        double SHC_rv = Function2(arg1, arg2);
    
        // post_call
        PyObject * SHTPy_rv = PyFloat_FromDouble(SHC_rv);
    
        return (PyObject *) SHTPy_rv;
    }

The return value of the function is converted into a ``PyObject``
in the *post_call* section of the wrapper.


Bool
^^^^

``PyArg_ParseTupleAndKeywords`` did not support boolean directly
until version 3.3. To deal with older versions of Python a ``PyObject``
is taken from the arguments then converted into a bool 
with ``PyObject_IsTrue`` during the *pre_call* phase.

A simple C++ function which accepts and returns a ``bool`` argument::

    bool Function3(bool arg)
    {
        return ! arg;
    }

Added to the YAML file as before::

    functions:
    - decl: bool Function3(bool arg)

This will produce the wrapper::

    static PyObject *
    PY_Function3(
      PyObject *SHROUD_UNUSED(self),
      PyObject *args,
      PyObject *kwds)
    {
        PyObject * SHPy_arg;
        const char *SHT_kwlist[] = {
            "arg",
            NULL };
    
        if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!:Function3",
            const_cast<char **>(SHT_kwlist), &PyBool_Type, &SHPy_arg))
            return NULL;
    
        // pre_call
        bool arg = PyObject_IsTrue(SHPy_arg);
    
        bool SHC_rv = Function3(arg);
    
        // post_call
        PyObject * SHTPy_rv = PyBool_FromLong(SHC_rv);
    
        return (PyObject *) SHTPy_rv;
    }


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

  - decl: void Sum(int  len,   +implied(size(values)),
                   int *values +dimension(:)+intent(in),
                   int *result +intent(out))

The ``dimension`` attribute defines the variable as a one dimensional
array.  NumPy is used to create an array from the argument
to the Python function. C pointers have no
idea how many values they point to.  This information is passed
by the *len* argument.

The *len* argument defines the ``implied`` attribute.  This argument
is not part of the Python API since its presence is *implied* from the
expression ``size(values)``. This uses the NumPy
to compute the total number of elements in the array.  It then passes
this value to the C wrapper::

    static PyObject *
    PY_Sum(
      PyObject *SHROUD_UNUSED(self),
      PyObject *args,
      PyObject *kwds)
    {
        PyObject * SHTPy_values;
        PyArrayObject * SHPy_values = NULL;
        const char *SHT_kwlist[] = {
            "values",
            NULL };
    
        if (!PyArg_ParseTupleAndKeywords(args, kwds, "O:Sum",
            const_cast<char **>(SHT_kwlist), &SHTPy_values))
            return NULL;
    
        // post_parse
        SHPy_values = reinterpret_cast<PyArrayObject *>(PyArray_FROM_OTF(
            SHTPy_values, NPY_INT, NPY_ARRAY_IN_ARRAY));
        if (SHPy_values == NULL) {
            PyErr_SetString(PyExc_ValueError,
                "values must be a 1-D array of int");
            goto fail;
        }
        {
            // pre_call
            int * values = static_cast<int *>(PyArray_DATA(SHPy_values));
            int result;  // intent(out)
            int len = PyArray_SIZE(SHPy_values);
    
            Sum(len, values, &result);
    
            // post_call
            PyObject * SHPy_result = PyInt_FromLong(result);
    
            // cleanup
            Py_DECREF(SHPy_values);
    
            return (PyObject *) SHPy_result;
        }
    
    fail:
        Py_XDECREF(SHPy_values);
        return NULL;
    }


Character
^^^^^^^^^

C++ ``std::string`` variables must be created from the NULL-terminated
strings returned by ``PyArg_ParseTupleAndKeywords``.

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

The Fortran wrapper requires the ``+len(30)`` attribute.
The Python wrapper will ignore this attribute.
The contents of the ``std::string`` result from the function
are copied into a Python object and returned to the user.

.. talk about memory leak

Attributes may also be added by assign new fields in **attrs**::

    - decl: const std::string Function4a(
        const std::string& arg1,
        const std::string& arg2 )
      attrs:
        result:
          len: 30

The wrapped function is::

    static PyObject *
    PY_Function4a(
      PyObject *SHROUD_UNUSED(self),
      PyObject *args,
      PyObject *kwds)
    {
        const char * arg1;
        const char * arg2;
        const char *SHT_kwlist[] = {
            "arg1",
            "arg2",
            NULL };
    
        if (!PyArg_ParseTupleAndKeywords(args, kwds, "ss:Function4a",
            const_cast<char **>(SHT_kwlist), &arg1, &arg2))
            return NULL;
    
        // post_parse
        const std::string SH_arg1(arg1);
        const std::string SH_arg2(arg2);
    
        const std::string SHCXX_rv = Function4a(SH_arg1, SH_arg2);
    
        // post_call
        PyObject * SHTPy_rv = PyString_FromString(SHCXX_rv.c_str());
    
        return (PyObject *) SHTPy_rv;
    }

The function is called as::

     >>> tutorial.Function4a("dog", "cat")
     'dogcat'

.. note :: This function is just for demonstration purposes.
           Any reasonable person would just add the strings together.

Default Value Arguments
------------------------

Each function with default value arguments will create a wrapper which
checks the number of arguments, then calls the function appropriately::

    double Function5(double arg1, bool arg2)
    {
        if (arg2) {
            return arg1 + 10.0;
        } else {
            return arg1;
        }
     }

Describe the function in YAML::

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

    static PyObject *
    PY_Function5_arg1_arg2(
      PyObject *SHROUD_UNUSED(self),
      PyObject *args,
      PyObject *kwds)
    {
        Py_ssize_t SH_nargs = 0;
        double arg1;
        PyObject * SHPy_arg2;
        const char *SHT_kwlist[] = {
            "arg1",
            "arg2",
            NULL };
        double SHC_rv;
    
        if (args != NULL) SH_nargs += PyTuple_Size(args);
        if (kwds != NULL) SH_nargs += PyDict_Size(args);
        if (!PyArg_ParseTupleAndKeywords(args, kwds, "|dO!:Function5",
            const_cast<char **>(SHT_kwlist), &arg1, &PyBool_Type,
            &SHPy_arg2))
            return NULL;
        switch (SH_nargs) {
        case 0:
            SHC_rv = Function5();
            break;
        case 1:
            SHC_rv = Function5(arg1);
            break;
        case 2:
            {
                // pre_call
                bool arg2 = PyObject_IsTrue(SHPy_arg2);
    
                SHC_rv = Function5(arg1, arg2);
                break;
            }
        }
    
        // post_call
        PyObject * SHTPy_rv = PyFloat_FromDouble(SHC_rv);
    
        return (PyObject *) SHTPy_rv;
    }

Python usage::

        >>> tutorial.Function5()
        13.1415
        >>> tutorial.Function5(1.0)
        11.0
        >>> tutorial.Function5(1.0, False)
        1.0

.. note :: This will cause a problem when called with keyword arguments
           since arguments can be skipped.

           >>> tutorial.Function5(arg2=False)


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

.. Generic Functions is only needed for Fortran.


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

Each class is wrapped in an extension type which holds a
pointer to an C++ instance of the class.  Class
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

The file ``pyTutorialmodule.hpp`` will have a struct for the class::

    typedef struct {
    PyObject_HEAD
        Class1 * obj;
    } PY_Class1;

And the class is defined in the module initialization function::

    PY_Class1_Type.tp_new   = PyType_GenericNew;
    PY_Class1_Type.tp_alloc = PyType_GenericAlloc;
    if (PyType_Ready(&PY_Class1_Type) < 0)
        return RETVAL;
    Py_INCREF(&PY_Class1_Type);
    PyModule_AddObject(m, "Class1", (PyObject *)&PY_Class1_Type);

old::

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

    static PyObject *
    PY_class1_Method1(
      PY_Class1 *self,
      PyObject *SHROUD_UNUSED(args),
      PyObject *SHROUD_UNUSED(kwds))
    {
        self->obj->Method1();
        Py_RETURN_NONE;
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

Class static methods
^^^^^^^^^^^^^^^^^^^^

C++ class static methods are supported as Python class static method.
To wrap the method::

    class Singleton {
        static Singleton& getReference();
    }

Use the YAML input::

    - name: Singleton
      functions:
      - decl: static Singleton& getReference()

This adds the ``METH_STATIC`` flags into the PyMethodsDef description
of the function.  It can then be called from Python as a method on the class::

        obj0 = tutorial.Singleton.getReference()

