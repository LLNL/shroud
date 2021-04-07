.. Copyright (c) 2017-2021, Lawrence Livermore National Security, LLC and
   other Shroud Project Developers.
   See the top-level COPYRIGHT file for details.

   SPDX-License-Identifier: (BSD-3-Clause)

Fortran Previous Work
=====================

Communicating between languages has a long history.

Babel
-----

.. https://computation.llnl.gov/casc/components

https://computation.llnl.gov/projects/babel-high-performance-language-interoperability
Babel parses a SIDL (Scientific Interface Definition Language) file to
generate source. It is a hub-and-spokes approach where each language
it supports is mapped to a Babel runtime object.  The last release was
2012-01-06. http://en.wikipedia.org/wiki/Babel_Middleware

Chasm
-----

http://chasm-interop.sourceforge.net/ - This page is dated July 13, 2005

Chasm is a tool to improve C++ and Fortran 90 interoperability. Chasm
parses Fortran 90 source code and automatically generates C++ bridging
code that can be used in C++ programs to make calls to Fortran
routines. It also automatically generates C structs that provide a
bridge to Fortran derived types. Chasm supplies a C++ array descriptor
class which provides an interface between C and F90 arrays. This
allows arrays to be created in one language and then passed to and
used by the other
language. http://www.cs.uoregon.edu/research/pdt/users.php


 * `CHASM: Static Analysis and Automatic Code Generation for Improved Fortran 90 and C++ Interoperability <http://permalink.lanl.gov/object/tr?what=info:lanl-repo/lareport/LA-UR-01-4955>`_ 
    C.E. Rasmussen, K.A. Lindlan, B. Mohr, J. Striegnitz

 * `Bridging the language gap in scientific computing: the Chasm approach <https://onlinelibrary.wiley.com/doi/abs/10.1002/cpe.909>`_ C. E. Rasmussen, M. J. Sottile, S. S. Shende, A. D. Malony (2005)

wrap
----

https://github.com/scalability-llnl/wrap

a PMPI wrapper generator

Trilinos
--------

http://trilinos.org/

Trilonos wraps C++ with C, then the Fortran over the C.  Described in the book Scientific Software Design. http://www.amazon.com/Scientific-Software-Design-The-Object-Oriented/dp/0521888131

  * `On the object-oriented design of reference-counted shadow objects <https://dl.acm.org/citation.cfm?doid=1985782.1985786>`_ Karla Morris, Damian W.I. Rouson, Jim Xia (2011)
  * `This Isn't Your Parents' Fortran: Managing C++ Objects with Modern Fortran <http://ieeexplore.ieee.org/document/6159199>`_ Damian Rouson, Karla Morris, Jim Xia (2012)


Directory packages/ForTrilinos/src/skeleton has a basic template which must be edited to create a wrapper for a class.


Exascale Programming: Adapting What We Have Can (and Must) Work

    In 2009 and 2010, the C++ based Trilinos project developed Fortran
    interface capabilities, called ForTrilinos. As an object-oriented (OO)
    collection of libraries, we assumed that the OO features of Fortran
    2003 would provide us with natural mappings of Trilinos classes into
    Fortran equivalents. Over the two-year span of the ForTrilinos effort,
    we discovered that compiler support for 2003 features was very
    immature. ForTrilinos developers quickly came to know the handful of
    compiler developers who worked on these features and, despite close
    collaboration with them to complete and stabilize the implementation
    of Fortran 2003 features (in 2010), ForTrilinos stalled and is no
    longer developed.

http://www.hpcwire.com/2016/01/14/24151/

https://github.com/Trilinos/ForTrilinos
https://www.researchgate.net/project/ForTrilinos

This is the new effort to provide Fortran interfaces to Trilinos
through automatic code generation using SWIG. The previous effort
(ca. 2008-2012) can be obtained by downloading Trilinos releases prior
to 12.12.

https://trilinos.github.io/ForTrilinos/files/ForTrilinos_Design_Document.pdf

SWIG
----

The custom version of swig available at https://github.com/swig-fortran/swig

.. The custom version of swig available at https://github.com/sethrj/swig

http://www.icl.utk.edu/~luszczek/conf/2019/siam_cse/siam-cse-johnsonsr.pdf
https://info.ornl.gov/sites/publications/Files/Pub127965.pdf

MPICH
-----

MPICH uses a custom perl scripts which has routine names and types in the source.

http://git.mpich.org/mpich.git/blob/HEAD:/src/binding/fortran/use_mpi/buildiface

GTK
---

gtk-fortran uses a python script which grep the C source to generate the Fortran.

https://github.com/jerryd/gtk-fortran/blob/master/src/cfwrapper.py
https://github.com/vmagnin/gtk-fortran/wiki

CDI
---

CDI is a C and Fortran Interface to access Climate and NWP model Data. https://code.zmaw.de/projects/cdi

"One part of CDI[1] is a such generator. It still has some rough edges and we haven't yet decided what to do about functions returning char * (it seems like that will need some wrapping unless we simply return TYPE(c_ptr) and let the caller deal with that) but if you'd like to have a starting point in Ruby try interfaces/f2003/bindGen.rb from the tarball you can download" https://groups.google.com/d/msg/comp.lang.fortran/oadwd3HHtGA/J8DD8kGeVw8J

Forpy
-----

This is a Fortran interface over the Python API written using the metaprogramming tool Fypp.

  * `Forpy: A library for Fortran-Python interoperability <https://github.com/ylikx/forpy>`_ 
  * `Fypp — Python powered Fortran metaprogramming <https://github.com/aradi/fypp>`_

CNF
---

http://www.starlink.ac.uk/docs/sun209.htx/sun209.html

The CNF package comprises two sets of software which ease the task of
writing portable programs in a mixture of FORTRAN and C. F77 is a set
of C macros for handling the FORTRAN/C subroutine linkage in a
portable way, and CNF is a set of functions to handle the difference
between FORTRAN and C character strings, logical values and pointers
to dynamically allocated memory.

h2m-AutoFortran
---------------

https://github.com/Kaiveria/h2m-Autofortran-Tool

The h2m-AutoFortran tool is designed to allow easy calls to C
routines from Fortran programs. Given a header file in standard C,
h2m will produce a Fortran module providing function interfaces
which maintain interoperability with C. Features for which there
are no Fortran equivalents will not be translated and warnings 
will be written to standard error.
The h2m-AutoFortran tool is built into Clang, the LLVM C compiler.
During translation, the Clang abstract syntax tree (AST) is used to 
assemble information about the header file. 


Links
-----

  * `Technical Specification ISO/IEC TS 29113:2012 <http://www.iso.org/iso/iso_catalogue/catalogue_tc/catalogue_detail.htm?csnumber=45136>`_
  * `Generating C Interfaces <http://fortranwiki.org/fortran/show/Generating+C+Interfaces>`_
  * `Shadow-object interface between Fortran95 and C++ <http://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=753048>`_  Mark G. Gray, Randy M. Roberts, and Tom M. Evans (1999)
  * `Generate C interface from C++ source code using Clang libtooling <http://samanbarghi.com/blog/2016/12/06/generate-c-interface-from-c-source-code-using-clang-libtooling/>`_
  * `Memory leaks in derived types revisited <https://dl.acm.org/citation.cfm?id=962183>`_ G. W. Stewart (2003)
  * `A General Approach to Creating Fortran Interface for C++ Application Libraries <https://link.springer.com/chapter/10.1007/3-540-27912-1_14>`_

..  https://link.springer.com/content/pdf/10.1007%2F3-540-27912-1_14.pdf



.. other shroud https://dthompson.us/projects/shroud.html
   Shroud is a simple secret manager with a command line interface.
