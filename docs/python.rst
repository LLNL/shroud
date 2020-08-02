.. Copyright (c) 2017-2020, Lawrence Livermore National Security, LLC and
   other Shroud Project Developers.
   See the top-level COPYRIGHT file for details.

   SPDX-License-Identifier: (BSD-3-Clause)

Python
=======

.. note:: Work in progress

This section discusses Python specific wrapper details.


.. struct-as-class
   Each struct extension type will have some additional PyObjects added to control memory
   PY_member_object - An object which caches the user visible object and contains
     a pointer to the data.  For example, NumPy, array, struct
   PY_member_data - An object which contains the memory and how to destroy it.
        PyCapsule - memory converted by a list
        Byte, String (python2) - String object.
   In addition, the memory can be modified by library so do not
   cache PY_member_object. Instead recreate it each time.

   With NumPy ``struct.array is struct.array``.  Each time the getter is called, the same
   cached object is returned. This works because the object contains a pointer to the C memory.
   Modifiying the NumPy array also changes the C memory and vice versa.
   Should also work with Python array, bytesarray, struct types.
   A field like `char *` does not use value.obj since C can change the memory and the object
   will not be changed since strings are not mutable.
     

Wrapper
-------


Types
-----

type fields
-----------

PY_build_arg
^^^^^^^^^^^^

Argument for Py_BuildValue.  Defaults to *{cxx_var}*.
This field can be used to turn the argument into an expression such as
*(int) {cxx_var}*  or *{cxx_var}{cxx_member}c_str()*
*PY_build_format* is used as the format:: 

    Py_BuildValue("{PY_build_format}", {PY_build_arg});

PY_build_format
^^^^^^^^^^^^^^^

'format unit' for Py_BuildValue.
If *None*, use value of *PY_format*.
Defaults to *None*

PY_format
^^^^^^^^^

'format unit' for PyArg_Parse and Py_BuildValue.
Defaults to *O*

PY_PyTypeObject
^^^^^^^^^^^^^^^

Variable name of PyTypeObject instance.
Defaults to *None*.

PY_PyObject
^^^^^^^^^^^

Typedef name of PyObject instance.
Defaults to *None*.

PY_ctor
^^^^^^^

Expression to create object.
ex. ``PyInt_FromLong({rv})``
Defaults to *None*.

PY_get
^^^^^^

Expression to get value from an object.
ex. ``PyInt_AsLong({py_var})``
Defaults to *None*.

PY_to_object_idtor
^^^^^^^^^^^^^^^^^^

Create an Python object for the type.
Includes the index of the destructor function.
Used with structs/classes that are created by functions
and must be wrapped.
``object = converter(address, idtor)``.
Defaults to *None*.

PY_to_object
^^^^^^^^^^^^

PyBuild - ``object = converter(address)``.
Defaults to *None*.

PY_from_object
^^^^^^^^^^^^^^

PyArg_Parse - ``status = converter(object, address)``.
Defaults to *None*.

py_type
^^^^^^^

The type returned by *PY_get* function.
Defaults to ``None`` which implies it is the same as the typemap.
i.e. ``PyInt_AsLong`` returns a ``long``.

Defined for complex types because ``PyComplex_AsCComplex`` returns
type ``Py_complex``.
See also *pytype_to_pyctor* and *pytype_to_cxx*.

pytype_to_pyctor
^^^^^^^^^^^^^^^^

Expression to use with *PY_ctor*.
Defaults to ``None`` which indicates no additional processing of the argument
is required.
Only needs to be defined when *py_type* is defined.

With complex types, it is used to extract the real and imaginary parts from
``Py_complex`` (defined with *py_type*)
with ``creal({ctor_expr}), cimag({ctor_expr})``.
*ctor_expr* is the expression used with *Py_ctor*.

pytype_to_cxx
^^^^^^^^^^^^^

Expression to convert *py_type* into a C++ value.
Only needs to be defined when *py_type* is defined.

Used with complex to convert ``Py_complex``  (defined with *py_type*)
to C using ``{work_var}.real + {work_var}.imag * I``
or C++ with ``std::complex(\tcvalue.real, cvalue.imag)``.

PYN_descr
^^^^^^^^^

Name of ``PyArray_Descr`` variable which describe type.
Used with structs.
Defaults to *None*.

PYN_typenum
^^^^^^^^^^^

NumPy type number.
ex. ``NPY_INT``
Defaults to *None*.


Statements
----------

The template for a function is:

.. code-block:: text

    static char {PY_name_impl}__doc__[] = "{PY_doc_string}";

    static PyObject *'
    {PY_name_impl}(
        {PY_PyObject} *{PY_param_self},
        PyObject *{PY_param_args},
        PyObject *{PY_param_kwds})
    {
        {declare}

        // {parse_format}  {parse_args}
        if (!PyArg_ParseTupleAndKeywords(
            {PY_param_args}, {PY_param_kwds}, "{PyArg_format}",
            SH_kw_list, {PyArg_vargs})) {
            return NULL;
        }

        // result pre_call
        
        // Create C from Python objects
        // Create C++ from C
        {post_parse}
        {               create scope before fail
          {pre_call}    pre_call declares variables for arguments

          call  {arg_call}
          {post_call}

          per argument
            // Create Python object from C++
            {ctor}    {post_call}

            {PyObject} *  {py_var} Py_BuildValue("{Py_format}", {vargs});
            {cleanup}
         }
         return;

       fail:
          {fail}
          Py_XDECREF(arr_x);
    }


The template for a setter is:

.. code-block:: text

    static PyObject *{PY_getter}(
        {PY_PyObject} *{PY_param_self},
        void *SHROUD_UNUSED(closure)) {
        {setter}
    }

The template for a getter is:

.. code-block:: text

    static int {PY_setter}("
        {PY_PyObject} *{PY_param_self},
        PyObject *{py_var},
        void *SHROUD_UNUSED(closure)) {
        {getter}
        return 0;
    }


Fields listed in the order they generate code.
C variables are created before the call to ``Py_ParseArgs``.
C++ variables are then created in *post_parse* and *pre_call*.
For example, creating a ``std::string`` from a ``char *``.

allocate_local_var
^^^^^^^^^^^^^^^^^^

Functions which return a struct/class instance (such as std::vector)
need to allocate a local variable which will be used to store the result.
The Python object will maintain a pointer to the instance until it is
deleted.

c_header
^^^^^^^^

cxx_header
^^^^^^^^^^

c_helper
^^^^^^^^

Blank delimited list of helper functions required for the wrapper.
The name may contain format strings and will be expand before it is
used.  ex. ``to_PyList_{cxx_type}``.
The function associated with the helper will be named *hnamefunc0*,
*hnamefunc1*, ... for each helper listed.

need_numpy
^^^^^^^^^^

If *True*, add NumPy headers and initialize in the module.

fmtdict
^^^^^^^

Update format dictionary to override generated values.
Each field will be evaluated before assigment.


ctor_expr - Expression passed to Typemap.PY_ctor
``PyInt_FromLong({ctor_expr})``.
Useful to add dereferencing if necessary.
``PyInt_FromLong`` is from typemap.PY_ctor.

.. code-block:: python

        fmtdict=dict(
            ctor_expr="{c_var}",
        ),


arg_declare
^^^^^^^^^^^

By default a local variable will be declared the same type as the
argument to the function.

For some cases, this will not be correct.  This field will be used
to replace the default declaration.

references

In some cases the declaration is correct but need to be initialized.
For example, setting a pointer.

Assign a blank list will not add any declarations.
This is used when only an output ``std::string`` or ``std::vector``
is created after parsing arguments.

This variables is used with ``PyArg_ParseTupleAndKeywords``.

The argument will be non-const to allow it to be assigned later.

.. code-block:: python

        name="py_char_*_out_charlen",
        arg_declare=[
            "{c_const}char {c_var}[{charlen}];  // intent(out)",
        ],

declare
^^^^^^^

Code needed to declare local variable.
Often used to define variables of type ``PyObject *``.

.. When defined, *typemap.PY_format* is append to the
   format string for ``PyArg_ParseTupleAndKeywords`` and
   *c_var* is used to hold the parsed.

cxx_local_var
^^^^^^^^^^^^^

Set when a C++ variable is created by post_parse.
*scalar*

Used to set format fields *cxx_member*

parse_format
^^^^^^^^^^^^

Works together with *parse_args* to describe how to parse
``PyObject`` in ``PyArg_ParseTupleAndKeywords``.
*parse_format* is used in the *format* arguments and
*parse_args* is append to the call as a vararg.

.. code-block:: c

    int PyArg_ParseTupleAndKeywords(PyObject *args, PyObject *kw,
        const char *format, char *keywords[], ...)

The simplest use is to pass the object directly through so that it
can be operated on by *post_parse* or *pre_call* to convert the object
into a C/C++ variable. For example, convert a ``PyObject`` into
an ``int *``.

.. code-block:: python

    parse_format="O",
    parse_args=["&{pytmp_var}"],
    declare=[
        "PyObject * {pytmp_var};",
    ],

The format field *pytmp_var* is created by Shroud, but must be
declared if it is used.

It can also be used to provide a *converter* function which converts
the object:

.. code-block:: python

    parse_format="O&",
    parse_args=["{hnamefunc0}", "&{py_var}"],

From the Python manual:
Note that any Python object references which are provided to the
caller (of `PyArg_Parse`) are borrowed references; do not decrement
their reference count!

parse_args
^^^^^^^^^^

A list of wrapper variables that are passed to ``PyArg_ParseTupleAndKeywords``.
Used with *parse_format*.

cxx_local_var
^^^^^^^^^^^^^

Set to *scalar* or *pointer* depending on the declaration in *post_declare*
*post_parse* or *pre_call*.

post_declare
^^^^^^^^^^^^

Declaration of C++ variables after calling
``PyArg_ParseTupleAndKeywords``.
Usually involves object constructors such as ``std::string`` or ``std::vector``.
Or for extracting struct and class pointers out of a `PyObject`.

These declarations should not include ``goto fail``.
This allows them to be created without a
"jump to label 'fail' crosses initialization of" error.

"It is possible to transfer into a block, but not in a way that
bypasses declarations with initialization. A program that jumps from a
point where a local variable with automatic storage duration is not in
scope to a point where it is in scope is ill-formed unless the
variable has POD type (3.9) and is declared without an initializer."

post_parse
^^^^^^^^^^
Statements to execute after the call to ``PyArg_ParseTupleAndKeywords``.
Used to convert C values into C++ values:

.. code-block:: text

    {var} = PyObject_IsTrue({var_obj});

Will not be added for class constructor objects.
since there is no need to build return values.


Allow *intent(in)* arguments to be processed.
For example, process ``PyObject`` into ``PyArrayObject``.

pre_call
^^^^^^^^

Location to allocate memory for output variables.
All *intent(in)* variables have been processed by *post_parse* so
their lengths are known.

arg_call
^^^^^^^^

List of arguments to pass to function.

post_call
^^^^^^^^^

Convert result and *intent(out)* into ``PyObject``.
Set *object_created* to True if a ``PyObject`` is created.


cleanup
^^^^^^^

Code to remove any intermediate variables.

fail
^^^^

Code to remove allocated memory and created objects.

goto_fail
^^^^^^^^^

If *True*, one of the other blocks such as *post_parse*, *pre_call*,
and *post_call* contain a call to ``fail``.
If any statements block sets *goto_fail*, then the *fail* block will
be inserted into the code/

.. object conversion


object_created
^^^^^^^^^^^^^^

Set to ``True`` when a ``PyObject`` is created by *post_call*.
This prevents ``Py_BuildValue`` from converting it into an Object.
For example, when a pointer is converted into a ``PyCapsule`` or
when NumPy is used to create an object.


Predefined Types
----------------

Int
^^^
An ``int`` argument is converted to Python with the typemap:

.. code-block:: yaml

    type: int
    fields:
        PY_format: i
        PY_ctor: PyInt_FromLong({c_deref}{c_var})
        PY_get: PyInt_AsLong({py_var})
        PYN_typenum: NPY_INT

Pointers
--------

When a function returns a pointer to a POD type several Python
interfaces are possible. When a function returns an ``int *`` the
simplest result is to return a ``PyCapsule``.  This is just the raw
pointer returned by C++.  It's also the least useful to the caller
since it cannot be used directly.
The more useful option is to assume that the result is a pointer to a scalar.
In this case a NumPy scalar can be returned or a Python object such 
as ``int`` or ``float``.

If the C++ library function can also provide the length of the
pointer, then its possible to return a NumPy array.
If *owner(library)* is set, the memory will never be released.
If *owner(caller)* is set, the the memory will be released when the
object is deleted.

The argument ``int *result+intent(OUT)+dimension(3)`` will create a
NumPy array, then pass the pointer to the data to the C function which
will presumably fill the contents.  The NumPy array will be returned
as part of the function result.  The dimension attribute must specify
a length.


Class Types
-----------

An extension type is created for each C++ class:

.. code-block:: c++

    typedef struct {
    PyObject_HEAD
        {namespace_scope}{cxx_class} * {PY_obj};
    } {PY_PyObject};


Extension types
^^^^^^^^^^^^^^^

Additional type information can be provided in the YAML file to generate place
holders for extension type methods:

.. code-block:: yaml

  - name: ExClass2
    cxx_header: ExClass2.hpp
    python:
      type: [dealloc, print, compare, getattr, setattr,
             getattro, setattro,
             repr, hash, call, str,
             init, alloc, new, free, del]

