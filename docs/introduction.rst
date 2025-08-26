.. Copyright Shroud Project Developers. See LICENSE file for details.

   SPDX-License-Identifier: (BSD-3-Clause)

Introduction
============

**Shroud** is a tool for creating a Fortran or Python interface to a C
or C++ library.  It can also create a C API for a C++ library.

The user creates a YAML file with the C/C++ declarations to be wrapped
along with some annotations to provide semantic information and code
generation options.  Shroud produces a wrapper for the library.
The generated code is highly-readable and intended to be similar to code
that would be hand-written to create the bindings.

Input is read from the YAML file which describes the types, variables,
enumerations, functions, structures and classes to wrap.  This file
must be created by the user.  Shroud does not parse C++ code to
extract the API. That was considered a large task and not needed for
the size of the API of the library that inspired Shroud's
development. In addition, there is a lot of semantic information which
must be provided by the user that may be difficult to infer from the
source alone.  However, the task of creating the input file is
simplified since the C++ declarations can be cut-and-pasted into the
YAML file.

In some sense, Shroud can be thought of as a fancy macro processor.
It takes the function declarations from the YAML file, breaks them
down into a series of contexts (library, class, function, argument)
and defines a dictionary of format macros of the form key=value.
There are then a series of macro templates which are expanded to
create the wrapper functions. The overall structure of the generated
code is defined by the classes and functions in the YAML file as well
as the requirements of C++ and Fortran syntax.

Each declaration can have annotations which provide semantic
information.  This information is used to create more idiomatic
wrappers.  Shroud started as a tool for creating a Fortran wrapper
for a C++ library.  The declarations and annotations in the input file
also provide enough information to create a Python wrapper.

Goals
-----

  * Simplify the creating of wrapper for a C++ library.
  * Preserves the object-oriented style of C++ classes.
  * Create an idiomatic wrapper API from the C++ API.
  * Generate code which is easy to understand.
  * No dependent runtime library.

Fortran
-------

The Fortran wrapper is created by using the interoperability with C features
added in Fortran 2003.
This includes the ``iso_c_binding`` module and the ``bind`` and ``value`` keywords.
Fortran cannot interoperate with C++ directly and uses C as the lingua franca.
C++ can communicate with C via a common heritage and the ``extern "C"`` keyword.
A C API for the C++ API is produced as a byproduct of the Fortran wrapping.

Using a C++ API to create an object and call a method:

.. code-block:: c++

    Instance * inst = new Instance;
    inst->method(1);

In Fortran this becomes:

.. code-block:: fortran

    type(instance) inst
    inst = instance()
    call inst%method(1)

.. note :: The ability to generate C++ wrappers for Fortran is not supported.

Issues
^^^^^^

There is a long history of ad-hoc solutions to provide C and Fortran interoperability.
Any solution must address several problems:

  * Name mangling of externals.  This includes namespaces and operator overloading in C++.
  * Call-by-reference vs call-by-value differences
  * Length of string arguments.
  * Blank filled vs null terminated strings.

The 2003 Fortran standard added several features for interoperability with C:

  * iso_c_binding - intrinsic module which defines fortran kinds for matching with C's types.
  * ``BIND`` keyword to control name mangling of externals.
  * ``VALUE`` attribute to allow pass-by-value.

In addition, Fortran 2003 provides object oriented programming facilities:

   * Type extension
   * Procedure Polymorphism with Type-Bound Procedures
   * Enumerations compatible with C

*Further Interoperability of Fortran with C*, Technical Specification
TS 29113, now part of Fortran 2018, introduced additional features:

   * assumed-type
   * assumed-rank
   * ``ALLOCATABLE``, ``OPTIONAL``, and ``POINTER`` attributes may be
     specified for a dummy argument in a procedure interface that has
     the ``BIND`` attribute.
   * ``CONTIGUOUS`` attribute

.. A Fortran pointer is similar to a C++ instance in that it not only has
   the address of the memory but also contains meta-data such as the
   type, kind and shape of the array.  Some vendors document the struct
   used to store the metadata for an array.

   * GNU Fortran http://gcc.gnu.org/wiki/ArrayDescriptorUpdate
   * Intel 17 https://software.intel.com/en-us/node/678452
   * Intel 15.0 https://software.intel.com/en-us/node/525356

Shroud uses the features of Fortran 2003 as well as additional
generated code to solve the interoperability problem to create
an idiomatic interface.

Limitations
^^^^^^^^^^^

Not all features of C++ can be mapped to Fortran.  Variadic
function are not directly supported. Fortran supports ``OPTIONAL``
arguments but that does not map to variadic functions.  ``OPTIONAL``
has a known number of possible argument while variadic does not.

Templates will be explicitly instantiated.  The instances are listed
in the YAML file and a wrapper will be created for each one. However,
Fortran can not initantiate templates at compile time.

Lambda functions are not supported.

Some other features are not currently supported but will be in the future:
complex type, exceptions.

Requirements
^^^^^^^^^^^^

Fortran wrappers are generated as free-form source and require a Fortran 2003 compiler.
C code requires C99.

Python
------

The Python wrappers use the `CPython API <https://docs.python.org/3/c-api/index.html>`_
to create a wrapper for the library.

Requirements
^^^^^^^^^^^^

The generated code will require

* Python 2.7 or Python 3.4+
* NumPy can be used when using pointers with
  *rank*, *dimension* or *allocatable*, attributes.

XKCD
----

`XKCD 1319 <https://xkcd.com/1319>`_

.. image:: automation.png

`XKCD 974 <https://xkcd.com/974>`_

.. image:: the_general_problem.png

`XKCD 1205 <https://xkcd.com/1205>`_

.. image:: is_it_worth_the_time.png
