.. Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and
   other Shroud Project Developers.
   See the top-level COPYRIGHT file for details.

   SPDX-License-Identifier: (BSD-3-Clause)

Output
======

What files are created
----------------------

Shroud will create multiple output file which must be compiled with
C++ or Fortran compilers.

One C++ file will be created for the library and one file for each C++ class.
In addition a utility file will be created with routines which are
implemented in C but called from Fortran.  This includes some memory
management routines.

Fortran creates a file for the library and one per additional namespace.
Since Fortran does not support forward referencing of derived types,
it is necessary to add all classes from a namespace into a single module.

.. XXX some comment about submodules?

Each Fortran file will only contain one module to make it easier to
create makefile dependencies using pattern rules:

.. code-block:: makefile

    %.o %.mod : %.f

File names for the header and implementation files can be set
explicitly by setting variables in the format of the global or class scope:

.. code-block:: yaml

    format:
      C_header_filename: top.h
      C_impl_filename: top.cpp
      F_impl_filename: top.f

    declarations:
    - decl: class Names
      format:
        C_header_filename: foo.h
        C_impl_filename: foo.cpp
        F_impl_filename: foo.f
 

The default file names are controlled by global options.
The option values can be changed to avoid setting the name for 
each class file explicitly.
It's also possible to change just the suffix of files:

.. code-block:: yaml

    options:
        YAML_type_filename_template: {library_lower}_types.yaml

        C_header_filename_suffix: h
        C_impl_filename_suffix: cpp
        C_header_filename_library_template: wrap{library}.{C_header_filename_suffix}
        C_impl_filename_library_template: wrap{library}.{C_impl_filename_suffix}

        C_header_filename_namespace_template: wrap{file_scope}.{C_header_file_suffix}
        C_impl_filename_namespace_template: wrap{file_scope}.{C_impl_filename_suffix}

        C_header_filename_class_template: wrap{cxx_class}.{C_header_file_suffix}
        C_impl_filename_class_template: wrap{cxx_class}.{C_impl_filename_suffix}

        F_filename_suffix: f
        F_impl_filename_library_template: wrapf{library_lower}.{F_filename_suffix}
        F_impl_filename_namespace_template: wrapf{file_scope}.{F_filename_suffix}

A file with helper functions may also be created.
For C the file is named by the format field *C_impl_utility*.
It contains files which are implemented in C but are called from Fortran
via ``BIND(C)``.

How names are created
---------------------

Shroud attempts to provide user control of names while providing
reasonable defaults.
Each name is based on the library, class, function or argument name
in the current scope.  Most names have a template which may be used
to control how the names are generated on a global scale.  Many names
may also be explicitly specified by a field.

For example, a library has an ``initialize`` function which is
in a namespace.  In C++ it is called as:

.. code-block:: c++

  #include "library.hpp"

  library::initialize()

By default this will be a function in a Fortran module and 
can be called as:

.. code-block:: fortran

  use library

  call initialize

Since ``initialize`` is a rather common name for a function, it may 
be desirable to rename the Fortran wrapper to something more specific.
The name of the Fortran implementation wrapper can be changed
by setting *F_name_impl*:

.. code-block:: yaml

    library: library

    declarations:
    - decl: namespace library
      declarations:
      - decl: void initialize
        format:
          F_name_impl: library_initialize

To rename all functions, set the template in the toplevel *options*:

.. code-block:: yaml

    library: library

    options:
      F_name_impl_template: "{library}_{underscore_name}{function_suffix}"

    declarations:
    - decl: namespace library
      declarations:
      - decl: void initialize

C++ allows allows overloaded functions and will mangle the names
behind the scenes.  With Fortran, the mangling must be explicit. To
accomplish this Shroud uses the *function_suffix* format string.  By
default, Shroud will use a sequence number.  By explicitly setting
*function_suffix*, a more meaningful name can be provided:

.. example from tutorial.yaml
.. code-block:: yaml

  - decl: void Function6(const std::string& name)
    format:
      function_suffix: _from_name
  - decl: void Function6(int indx)
    format:
      function_suffix: _from_index

This will create the Fortran functions ``function6_from_name`` and
``function6_from_index``.  A generic interface named ``function6``
will also be created which will include the two generated functions.

Likewise, default arguments will produce several Fortran wrappers and
a generic interface for a single C++ function. The format dictionary
only allows for a single *function_default* per function.  Instead the
field *default_arg_suffix* can be set.  It contains a list of
*function_suffix* values which will be applied from the minimum to the
maximum number of arguments:

.. example from tutorial.yaml
.. code-block:: yaml

  - decl: int overload1(int num,
            int offset = 0, int stride = 1)
    default_arg_suffix:
    - _num
    - _num_offset
    - _num_offset_stride

Finally, multiple Fortran wrappers can be generated from a single
templated function. Each instantiation will generate an additional
Fortran Wrapper and can be distinguished by the *template_suffix*
entry of the format dictionary.

If there is a single template argument, then *template_suffix* will be
set to the *flat_name* field of the instantiated argument.  For
example, ``<int>`` defaults to ``_int``.  This works well for POD types.
The entire qualified name is used.  For ``<std::string>`` this would be
``std_string``.  Classes which are deeply nested can produce very long
values for *template_suffix*. To deal with this, the
*function_template* field can be set on Class declarations:

.. code-block:: yaml

    - decl: namespace internal
      declarations:
      - decl: class ImplWorker1
        format:
          template_suffix: instantiation3

By default ``internal_implworker1`` would be used for the
*template_suffix*.  But in this case ``instantiation3`` will be used.

For multiple template arguments, *template_suffix* defaults to a
sequence number to avoid long function names.  In this case,
specifying an explicit *template_suffix* can produce a more user
friendly name:

.. code-block:: yaml

    - decl: template<T,U> void FunctionTU(T arg1, U arg2)
      cxx_template:
      - instantiation: <int, long>
        format:
          template_suffix: instantiation1
      - instantiation: <float, double>
        format:
          template_suffix: instantiation2

The Fortran functions will be named ``function_tu_instantiation1`` and
 ``function_tu_instantiation2``.

Additional Wrapper Functions
----------------------------

Functions can be created in the Fortran wrapper which have no
corresponding function in the C++ library.  This may be necessary to
add functionality which may unnecessary in C++.  For example, a
library provides a function which returns a string reference to a
name.  If only the length is desired no extra function is required in
C++ since the length is extracted used a ``std::string`` method:

.. code-block:: c++

    ExClass1 obj("name")
    int len = obj.getName().length();

Calling the Fortran ``getName`` wrapper will copy the string into a
Fortran array but you need the length first to make sure there is
enough room.  You can create a Fortran wrapper to get the length
without adding to the C++ library:

.. code-block:: yaml

    declarations:
    - decl: class ExClass1
      declarations:
      - decl: int GetNameLength() const
        format:
          C_code: |
            {C_pre_call}
            return {CXX_this}->getName().length();

The generated C wrapper will use the *C_code* provided for the body:

.. code-block:: c++

    int AA_exclass1_get_name_length(const AA_exclass1 * self)
    {
        const ExClass1 *SH_this = static_cast<const ExClass1 *>(
            static_cast<const void *>(self));
        return SH_this->getName().length();
    }

The *C_pre_call* format string is generated by Shroud to convert the
``self`` argument into *CXX_this* and must be included in *C_code*
to get the definition.


.. Fortran shadow class

Helper functions
----------------

Shroud provides some additional file static function which are inserted 
at the beginning of the wrapped code. Some helper functions are used to
communicate between C and Fortran.  They are global and written into
the *fmt.C_impl_utility* file.  The names of these files will have
*C_prefix* prefixed to create unique names.

C helper functions

``ShroudStrCopy(char *dest, int ndest, const char *src, int nsrc)``
    Copy *src* into *dest*, blank fill to *ndest* characters
    Truncate if *dest* is too short to hold all of *src*.
    *dest* will not be NULL terminated.

``int ShroudLenTrim(const char *src, int nsrc)``
    Returns the length of character string *src* with length *nsrc*,
    ignoring any trailing blanks.

Each Python helper is prefixed by format variable *PY_helper_prefix* which
defaults to ``SHROUD_``.  This is used to avoid conflict with other
wrapped functions.

The option *PY_write_helper_in_util* will write all of the
helper fuctions into the file defined by *PY_utility_filename*.
This can be useful to avoid clutter when there are a lot of classes
which may create lots of duplicate helpers. The helpers will no longer
be file static and instead will also be prefixed with *C_prefix* to
avoid conflicting with helpers created by another Shroud wrapped library.


Header Files
^^^^^^^^^^^^

The header files for the library are included by the generated C++ source files.

The library source file will include the global *cxx_header* field.
Each class source file will include the class *cxx_header* field unless it is blank.
In that case the global *cxx_header* field will be used.

To include a file in the implementation list it in the global or class options:

.. code-block:: yaml

    cxx_header: global_header.hpp

    declarations:
    - decl: class Class1
      cxx_header: class_header.hpp

    - decl: typedef int CustomType
        c_header:  type_header.h
        cxx_header : type_header.hpp


The *c_header* field will be added to the header file of contains functions
which reference the type.
This is used for files which are not part of the library but which contain code
which helps map C++ constants to C constants

A global *fortran_header* field will insert ``#include`` lines to be
used with the Fortran preprocessor (typically a variant of the C
preprocessor).  This will work with the ``cpp_if`` lines in
declarations which will conditionally compile a wrapper.

.. FILL IN MORE

Local Variable
^^^^^^^^^^^^^^

*SH_* prefix on local variables which are created for a corresponding argument.
For example the argument `char *name`, may need to create a local variable
named `std::string SH_name`.

Shroud also generates some code which requires local variables such as
loop indexes.  These are prefixed with *SHT_*.  This name is controlled 
by the format variable *c_temp*.

Results are named from *fmt.C_result* or *fmt.F_result*.

Format variable which control names are

* c_temp
* C_local
* C_this
* CXX_local
* CXX_this
* C_result

* F_result - ``SHT_rv``  (return value)
* F_this - ``obj``

* LUA_result

* PY_result


C Preprocessor
--------------

It is possible to add C preprocessor conditional compilation
directives to the generated source.  For example, if a function should
only be wrapped if ``USE_MPI`` is defined the ``cpp_if`` field can be
used:

.. code-block:: yaml

    - decl: void testmpi(MPI_Comm comm)
      format:
        function_suffix: _mpi
      cpp_if: ifdef HAVE_MPI
    - decl: void testmpi()
      format:
        function_suffix: _serial
      cpp_if: ifndef HAVE_MPI

The function wrappers will be created within ``#ifdef``/``#endif``
directives.  This includes the C wrapper, the Fortran interface and
the Fortran wrapper.  The generated Fortran interface will be:

.. code-block:: fortran

        interface testmpi
    #ifdef HAVE_MPI
            module procedure testmpi_mpi
    #endif
    #ifndef HAVE_MPI
            module procedure testmpi_serial
    #endif
        end interface testmpi

Class generic type-bound function will also insert conditional
compilation directives:

.. code-block:: yaml

    - decl: class ExClass3
      cpp_if: ifdef USE_CLASS3
      declarations:
      - decl: void exfunc()
        cpp_if: ifdef USE_CLASS3_A
      - decl: void exfunc(int flag)
        cpp_if: ifndef USE_CLASS3_A

The generated type will be:

.. code-block:: fortran

        type exclass3
            type(SHROUD_capsule_data), private :: cxxmem
        contains
            procedure :: exfunc_0 => exclass3_exfunc_0
            procedure :: exfunc_1 => exclass3_exfunc_1
    #ifdef USE_CLASS3_A
            generic :: exfunc => exfunc_0
    #endif
    #ifndef USE_CLASS3_A
            generic :: exfunc => exfunc_1
    #endif
        end type exclass3

A ``cpp_if`` field in a class will add a conditional directive around
the entire class.

Finally, ``cpp_if`` can be used with types. This would be required in
the first example since ``mpi.h`` should only be included when
``USE_MPI`` is defined:

.. code-block:: yaml

    typemaps:
    - type: MPI_Comm
      fields:
        cpp_if: ifdef USE_MPI


When using ``cpp_if``, it is useful to set the option
``F_filename_suffix`` to ``F``. This will cause most compilers to
process the Fortran souce with ``cpp`` before compilation.
The ``fortran_header`` field can be added to the YAML file to
insert ``#include`` directives at the top of the Fortran source files.

The ``typemaps`` field can only appear at the outermost layer
and is used to augment existing typemaps.


Debugging
---------

Shroud generates a JSON file with all of the input from the YAML
and all of the format dictionaries and type maps.
This file can be useful to see which format keys are available and
how code is generated.

