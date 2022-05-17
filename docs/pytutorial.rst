.. Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and
   other Shroud Project Developers.
   See the top-level COPYRIGHT file for details.

   SPDX-License-Identifier: (BSD-3-Clause)

:orphan:

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

The simplest item to wrap is a function in the file ``tutorial.hpp``:

.. code-block:: c++

   namespace tutorial {
     void Function1(void);
   }

This is wrapped using a YAML input file ``tutorial.yaml``:

.. code-block:: yaml

  library: Tutorial
  cxx_header: tutorial.hpp

  options:
    wrap_fortran: False
    wrap_c: False
    wrap_python: True
    debug: True

  declarations:
  - decl: namespace tutorial
    declarations:
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

Process the file with *Shroud*:

.. code-block:: sh

    % shroud tutorial.yaml
    Wrote pyClass1type.cpp
    Wrote pyTutorialmodule.hpp
    Wrote pyTutorialmodule.cpp
    Wrote pyTutorialhelper.cpp

The generated C function in file ``pyTutorialmodule.cpp`` is:

.. code-block:: c++

    static PyObject *
    PY_Function1(
      PyObject *SHROUD_UNUSED(self),
      PyObject *SHROUD_UNUSED(args),
      PyObject *SHROUD_UNUSED(kwds))
    {
        tutorial::Function1();
        Py_RETURN_NONE;
    }

The wrapper implementation function is named using the format **PY_name_impl**
which defaults to ``{PY_prefix}{class_prefix}{function_name}{function_suffix}``.
This can be changed on a global level by resetting the template option:

.. code-block:: yaml

    options:
      PY_name_impl_template: {PY_prefix}{class_prefix}{function_name}{function_suffix}_extra

All of the functions in the wrapper are file static exception for the module 
initialization function. To give them a unique name from the function which
is being wrapped the prefix **PY_format** is used.  It defaults to ``PY_``
but can be set to another value:

.. code-block:: yaml

    format:
      PY_prefix: NEW_

Since the prototype of the function is required by the Python API,
the ``SHROUD_UNUSED`` macro is used to help avoid some compiler errors
about unused arguments.

The function uses the macro ``Py_RETURN_NONE`` from the Python API
to indicate a successful executation that returns no values.

Some additional boiler plate is created for the function:

.. code-block:: c++

    static PyMethodDef PY_methods[] = {
        {"Function1", (PyCFunction)PY_Function1, METH_NOARGS,
            PY_Function1__doc__},
        {NULL,   (PyCFunction)NULL, 0, NULL}            /* sentinel */
    };

Finally the module creation function is added at the end of the file:

.. code-block:: c++

    extern "C" PyMODINIT_FUNC
    #ifdef PY_MAJOR_VERSION >= 3
    PyInit_tutorial(void)
    #else
    inittutorial(void)
    #endif
    {
        PyObject *m = NULL;
        const char * error_name = "tutorial.Error";
    
        /* Create the module and add the functions */
    #if PY_MAJOR_VERSION >= 3
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
To wrap ``Function2``:

.. code-block:: c++

    double Function2(double arg1, int arg2)
    {
        return arg1 + arg2;
    }

Add the declaration to the YAML file:

.. code-block:: yaml

    declarations:
    - decl: double Function2(double arg1, int arg2)

Local variables are created for the argument values.
There values are filled in by ``PyArg_ParseTupleAndKeywords``.
The generated function is:

.. code-block:: c++

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
    
        double SHC_rv = tutorial::Function2(arg1, arg2);
    
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

A simple C++ function which accepts and returns a ``bool`` argument:

.. code-block:: c++

    bool Function3(bool arg)
    {
        return ! arg;
    }

Added to the YAML file as before:

.. code-block:: yaml

    declarations:
    - decl: bool Function3(bool arg)

This will produce the wrapper:

.. code-block:: c++

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
    
        bool SHC_rv = tutorial::Function3(arg);
    
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
``result`` is an output scalar:

.. code-block:: c++

    void Sum(int len, const int *values, int *result)
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
Python:

.. code-block:: c++

  - decl: void Sum(int  len,   +implied(size(values)),
                   const int *values +rank(1),
                   int *result +intent(out))

The ``rank`` attribute defines the variable as a one dimensional
array.  NumPy is used to create an array from the argument
to the Python function. C pointers have no
idea how many values they point to.  This information is passed
by the *len* argument.

The *len* argument defines the ``implied`` attribute.  This argument
is not part of the Python API since its presence is *implied* from the
expression ``size(values)``. This uses NumPy
to compute the total number of elements in the array.  It then passes
this value to the C wrapper:

.. code-block:: c++

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
    
            tutorial::Sum(len, values, &result);
    
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


String
^^^^^^

A Python ``str`` type is similar to a C++ ``std::string``.
A C++ ``std::string`` variable is created from the NULL-terminated
string returned by ``PyArg_ParseTupleAndKeywords``.

C++ routine:

.. code-block:: c++

    const std::string Function4a(
        const std::string& arg1,
        const std::string& arg2)
    {
        return arg1 + arg2;
    }

YAML input:

.. code-block:: yaml

    declarations:
    - decl: const std::string Function4a+len(30)(
        const std::string& arg1,
        const std::string& arg2 )

The Fortran wrapper requires the ``+len(30)`` attribute.
The Python wrapper will ignore this attribute.
The contents of the ``std::string`` result from the function
are copied into a Python object and returned to the user.

.. talk about memory leak

Attributes may also be added by assign new fields in **attrs**:

.. code-block:: yaml

    - decl: const std::string Function4a(
        const std::string& arg1,
        const std::string& arg2 )
      attrs:
        result:
          len: 30

The wrapped function is:

.. code-block:: c++

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
    
        const std::string SHCXX_rv = tutorial::Function4a(SH_arg1, SH_arg2);
    
        // post_call
        PyObject * SHTPy_rv = PyString_FromString(SHCXX_rv.c_str());
    
        return (PyObject *) SHTPy_rv;
    }

The function is called as:

.. code-block:: python

     >>> tutorial.Function4a("dog", "cat")
     'dogcat'

.. note :: This function is just for demonstration purposes.
           Any reasonable person would just add the strings together.

Default Value Arguments
------------------------

Each function with default value arguments will create a wrapper which
checks the number of arguments, then calls the function appropriately.
A header file contains:

.. code-block:: c++

    double Function5(double arg1 = 3.1415, bool arg2 = true)

and the function is defined as:

.. code-block:: c++

    double Function5(double arg1, bool arg2)
    {
        if (arg2) {
            return arg1 + 10.0;
        } else {
            return arg1;
        }
     }

Describe the function in YAML:

.. code-block:: yaml

    declarations:
    - decl: double Function5(double arg1 = 3.1415, bool arg2 = true)
      default_arg_suffix:
      -  
      -  _arg1
      -  _arg1_arg2

The *default_arg_suffix* provides a list of values of
*function_suffix* for each possible set of arguments for the function.
In this case 0, 1, or 2 arguments. For Python, *default_arg_suffix* is ignored
since only one function is created.

C wrappers:

.. code-block:: c++

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
            SHC_rv = tutorial::Function5();
            break;
        case 1:
            SHC_rv = tutorial::Function5(arg1);
            break;
        case 2:
            {
                // pre_call
                bool arg2 = PyObject_IsTrue(SHPy_arg2);
    
                SHC_rv = tutorial::Function5(arg1, arg2);
                break;
            }
        }
    
        // post_call
        PyObject * SHTPy_rv = PyFloat_FromDouble(SHC_rv);
    
        return (PyObject *) SHTPy_rv;
    }

Python usage:

.. code-block:: python

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

C++ allows function names to be overloaded.  Python supports this 
directly since it is not strongly typed.  The Python wrapper will attempt to 
call each overload until it finds one which matches the arguments.

C++:

.. code-block:: c++

    void Function6(const std::string &name);
    void Function6(int indx);

By default the names are mangled by adding an index to the end. This
can be controlled by setting **function_suffix** in the YAML file:

.. code-block:: yaml

  declarations:
  - decl: void Function6(const std::string& name)
    function_suffix: _from_name
  - decl: void Function6(int indx)
    function_suffix: _from_index

Each overloaded function is wrapped as usual but are not added to the Python module.
Instead, an additional function is created:

.. code-block:: c++

    static PyObject *
    PY_Function6(
      PyObject *self,
      PyObject *args,
      PyObject *kwds)
    {
        Py_ssize_t SHT_nargs = 0;
        if (args != NULL) SHT_nargs += PyTuple_Size(args);
        if (kwds != NULL) SHT_nargs += PyDict_Size(args);
        PyObject *rvobj;
        if (SHT_nargs == 1) {
            rvobj = PY_Function6_from_name(self, args, kwds);
            if (!PyErr_Occurred()) {
                return rvobj;
            } else if (! PyErr_ExceptionMatches(PyExc_TypeError)) {
                return rvobj;
            }
            PyErr_Clear();
        }
        if (SHT_nargs == 1) {
            rvobj = PY_Function6_from_index(self, args, kwds);
            if (!PyErr_Occurred()) {
                return rvobj;
            } else if (! PyErr_ExceptionMatches(PyExc_TypeError)) {
                return rvobj;
            }
            PyErr_Clear();
        }
        PyErr_SetString(PyExc_TypeError, "wrong arguments multi-dispatch");
        return NULL;
    }

They can be used as:

.. code-block:: python

        import tutorial
        tutorial.Function6("name")
        tutorial.Function6(1)


Optional arguments and overloaded functions
-------------------------------------------

Overloaded function that have optional arguments can also be wrapped:

.. code-block:: yaml

  - decl: int overload1(int num,
            int offset = 0, int stride = 1)
  - decl: int overload1(double type, int num,
            int offset = 0, int stride = 1)

These routines can then be called as:

.. code-block:: python

    rv = tutorial.overload1(10)
    rv = tutorial.overload1(1., 10)

    rv = tutorial.overload1(10, 11, 12)
    rv = tutorial.overload1(1., 10, 11, 12)

Templates
---------

C++ template are handled by creating a wrapper for each instantiation 
of the function defined by the **cxx_template** field.

C++:

.. code-block:: c++

  template<typename ArgType>
  void Function7(ArgType arg)
  {
      return;
  }

YAML:

.. code-block:: yaml

  - decl: |
       template<typename ArgType>
       void Function7(ArgType arg)
    cxx_template:
    - instantiation: <int>
    - instantiation: <double>

This will create a Python wrapper for each value of *ArgType*, ``int``
and ``double`` and then a single which will call the other two in
sucession looking for input arguments which match.
This is similar to ``Function6``.

.. note:: fix RetType for Python

Likewise, the return type can be templated but in this case no
interface block will be generated since generic function cannot vary
only by return type.

C++:

.. code-block:: c++

  template<typename RetType>
  RetType Function8()
  {
      return 0;
  }

YAML:

.. code-block:: yaml

  - decl: template<typename RetType> RetType Function8()
    cxx_template:
    - instantiation: <int>
    - instantiation: <double>

C wrapper:

.. code-block:: c++

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
use of a type:

.. code-block:: c++

    typedef int TypeID;

    int typefunc(TypeID arg);

Shroud must be told about user defined types in the YAML file::

.. code-block:: yaml

    declarations:
    - decl: typedef int TypeID;

This will map the C++ type ``TypeID`` to the predefined type ``int``.
The C wrapper will use ``int``:

.. code-block:: c++

    int TUT_typefunc(int arg)
    {
        int SHT_rv = typefunc(arg);
        return SHT_rv;
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

The enum is defined in the YAML as:

.. code-block:: yaml

    declarations:
    - decl: |
          enum Color {
            RED,
            BLUE,
            WHITE
          };

Integer parameters are created for each value:

.. code-block:: python

    >>> tutorial.RED
    0
    >>> type(tutorial.RED)
    <type 'int'>

.. note:: This isn't fully equivalent to C's enumerations since you can
          assign to them as well.


Structure
^^^^^^^^^

Structures in C++ are accessed using Numpy.
For example, the C++ code:

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

This will add a varible to the module which can be used to create
instances of the struct:

.. code-block:: python

    >>> import tutorial
    >>> type(tutorial.struct1_dtype)
    <type 'numpy.dtype'>
    >>> tutorial.struct1_dtype
    dtype({'names':['ifield','dfield'], 'formats':['<i4','<f8'], 'offsets':[0,8], 'itemsize':16}, align=True)

    >>> import numpy as np
    >>> val = np.array((1, 2.5), dtype=tutorial.struct1_dtype)
    >>> val
    array((1,  2.5), 
          dtype={'names':['ifield','dfield'], 'formats':['<i4','<f8'], 'offsets':[0,8], 'itemsize':16, 'aligned':True})

.. note:: All fields must be defined in the YAML file in order to ensure that
          C++'s ``sizeof`` and NumPy's ``itemsize`` are the same.


A function which returns a struct value will create a NumPy scalar using the dtype.
A C++ function which initialized a struct can be written as:

.. code-block:: yaml

    - decl: struct1 returnStruct(int i, double d);

To use the function:

.. code-block:: python

    >>> val = tutorial.returnStruct(1, 2.5)
    >>> val
    array((1,  2.5), 
          dtype={'names':['ifield','dfield'], 'formats':['<i4','<f8'], 'offsets':[0,8], 'itemsize':16, 'aligned':True})
    >>> val['ifield']
    array(1, dtype=int32)
    >>> val['dfield']
    array(2.5)


Classes
-------

Each class is wrapped in an extension type which holds a
pointer to an C++ instance of the class.

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
      - decl: ~Class1() +name(delete)
      - decl: void Method1()

The constructor and destructor have no method name associated with
them. The constructor is called by the ``tp_init`` method of the type
and the destructor is called by ``tp_del``.

The file ``pyTutorialmodule.hpp`` will have a struct for the class:

.. code-block:: c++

    typedef struct {
    PyObject_HEAD
        Class1 * obj;
    } PY_Class1;

And the class is defined in the module initialization function:

.. code-block:: c++

    PY_Class1_Type.tp_new   = PyType_GenericNew;
    PY_Class1_Type.tp_alloc = PyType_GenericAlloc;
    if (PyType_Ready(&PY_Class1_Type) < 0)
        return RETVAL;
    Py_INCREF(&PY_Class1_Type);
    PyModule_AddObject(m, "Class1", (PyObject *)&PY_Class1_Type);


The C++ code to call the function:

.. code-block:: c++

    #include <tutorial.hpp>
    tutorial::Class1 *cptr = new tutorial::Class1();
    cptr->Method1();

And the Python version::

.. code-block:: python

    import tutorial
    cptr = tutoral.Class1()
    cptr.method1()

Class static methods
^^^^^^^^^^^^^^^^^^^^

C++ class static methods are supported as Python class static method.
To wrap the method:

.. code-block:: c++

    class Singleton {
        static Singleton& getReference();
    }

Use the YAML input:

.. code-block:: yaml

    - decl: class Singleton
      declarations:
      - decl: static Singleton& getReference()

This adds the ``METH_STATIC`` flags into the PyMethodsDef description
of the function.  It can then be called from Python as a method on the class:

.. code-block:: python

        obj0 = tutorial.Singleton.getReference()

