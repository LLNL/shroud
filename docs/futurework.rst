.. Copyright (c) 2017-2019, Lawrence Livermore National Security, LLC. 
..
.. Produced at the Lawrence Livermore National Laboratory 
..
.. LLNL-CODE-738041.
..
.. All rights reserved. 
..
.. This file is part of Shroud.
..
.. For details about use and distribution, please read LICENSE.
..
.. #######################################################################

Future Work
===========

  * complex
  * structures

  * pointers to pointers and in particular ``char **`` are not supported.
    An argument like ``Class **ptr+intent(out)`` does not work.
    Instead use a function which return a pointer to ``Class *``

  * reference counting and garbage collection


The copying of strings solves the blank-filled vs null-terminated
differences between Fortran and C and works well for many strings.
However, if a large buffer is passed, it may be desirable to avoid the
copy.

There is some initial work to support Python and Lua wrappers.


Possible Future Work
--------------------

Use a tool to parse C++ and extract info.

  * https://github.com/CastXML/CastXML
  * https://pypi.python.org/pypi/pygccxml
  * Wrapping C and C++ Libraries with CastXML | SciPy 2015 | Brad King, Bill Hoffman, Matthew McCormick https://www.youtube.com/watch?v=O2lBgtaDdyk&index=20&list=PLYx7XA2nY5Gcpabmu61kKcToLz0FapmHu
