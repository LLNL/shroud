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

Introduction
============

Input is read from a YAML file which describes the types, variables,
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
wrappers.  **Shroud** started as a tool for creating a Fortran wrapper
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

Using a C++ API to create an object and call a method::

    Instance * inst = new Instance;
    inst->method(1);

In Fortran this becomes::

    type(instance) inst
    inst = instance_new()
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

In addition, Fortran 2003 provides some object oriented programming facilities:

   * Type extension
   * Procedure Polymorphism with Type-Bound Procedures
   * Enumerations compatible with C

A Fortran pointer is similar to a C++ instance in that it not only has
the address of the memory but also contains meta-data such as the
type, kind and shape of the array.  Some vendors document the struct
used to store the metadata for an array.

   * GNU Fortran http://gcc.gnu.org/wiki/ArrayDescriptorUpdate
   * Intel 17 https://software.intel.com/en-us/node/678452

..   * Intel 15.0 https://software.intel.com/en-us/node/525356

Fortran provides a **pointer** and **allocatable** attributes which are not
directly supported by C.  Each vendor has their own pointer struct.
Eventually this will be supported in Fortran via the Further Interoperability of Fortran and C -
`Technical Specification ISO/IEC TS 29113:2012 <http://www.iso.org/iso/iso_catalogue/catalogue_tc/catalogue_detail.htm?csnumber=45136>`_


Requirements
^^^^^^^^^^^^

Fortran wrappers are generated as free-form source and require a Fortran 2003 compiler.

Python
------

The Python wrappers use the `CPython API <https://docs.python.org/3/c-api/index.html>`_
to create a wrapper for the library.

Requirements
^^^^^^^^^^^^

The generated code will require

* Python 2.7 or Python 3.4+
* NumPy will be required when using pointers with
  *dimension*, *allocatable*, or *mold* attributes.

XKCD
----

`XKCD <https://xkcd.com/1319>`_

.. image:: automation.png
