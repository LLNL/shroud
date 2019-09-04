.. Copyright (c) 2017-2019, Lawrence Livermore National Security, LLC and
   other Shroud Project Developers.
   See the top-level COPYRIGHT file for details.

   SPDX-License-Identifier: (BSD-3-Clause)

Namespaces
==========

Each library or class can be associated with a namespace:

.. code-block:: c++

    namespace one {
      namespace two {
         void function();

         namespace three {
           class Class1 {
           };
         }

         class Class2 {
         };
      } // namespace two
    } // namespace one

    class Class3 {
    };

The YAML file would look like:

.. code-block:: yaml

    declarations:
    - decl: namespace one
      declarations:
      - decl: namespace two
        declarations:
        - decl: void function();
        - decl: namespace three
          declarations:
          - class: Class1
        - class: Class2
    - class: Class3

If only one set of namespaces are used in a file, the ``namespace``
field can be used at the global level to avoid excessive indenting.
For example, if *Class3* was not wrapped then the file could be
written as:

.. code-block:: yaml

    namespace: one two
    declarations:
    - decl: void function();
    - decl: namespace three
      declarations:
      - class: Class1
    - class: Class2

Each namespace is translated into a Fortran or Python module.

Each namespace contributes to part of the C name via the format
field *namespace_scope*


namespace_scope - The full C++ qualified name.
  Delimited by ``::``.

C_name_scope - Does not include the toplevel *namespace* entries.
  used with C_name_template, C_enum_member_template


F_name_scope -
  Name items within a fortran module with class.

.. C_scope_name - Does not include the toplevel *namespace* entries.

class_prefix
  class prefix where a namespace is a module, but there may be 
  multiple classes in a namespace.

cxx_class - legal identifier
cxx_type - C++ identifier


options.flatten_namespace

