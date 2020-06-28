.. Copyright (c) 2017-2020, Lawrence Livermore National Security, LLC and
   other Shroud Project Developers.
   See the top-level COPYRIGHT file for details.

   SPDX-License-Identifier: (BSD-3-Clause)

Python Previous Work
====================

There a several available tools to creating a Python interface to a C or C++ library.

.. https://docs.python.org/3/library/struct.html
   https://legacy.python.org/dev/peps/pep-3118/  Revising the buffer protocol
   https://docs.python.org/3/c-api/buffer.html

Ctypes
------

* http://docs.python.org/lib/module-ctypes.html

Pros
^^^^

* No need for compiler.

Cons
^^^^

* Difficult wrapping C++ due to mangling and object ABI.

SWIG
----

* http://www.swig.org/


PyBindgen
---------

* https://github.com/gjcarneiro/pybindgen
* http://pybindgen.readthedocs.io/en/latest/

Cython
------

* http://cython.org
* https://cython.readthedocs.io/en/latest/


http://blog.kevmod.com/2020/05/python-performance-its-not-just-the-interpreter/

I ran Cython (a Python->C converter) on the previous benchmark, and it
runs in exactly the same amount of time: 2.11s. I wrote a simplified C
extension in 36 lines compared to Cython's 3600, and it too runs in
2.11s.
  

SIP
---

Sip was developed to create PyQt.

* https://www.riverbankcomputing.com/software/sip/intro

Shiboken
--------

Shiboken was developed to create PySide.

* https://wiki.qt.io/Qt_for_Python
* http://doc.qt.io/qtforpython/shiboken2/contents.html


Boost Python
------------

* https://www.boost.org/doc/libs/1_66_0/libs/python/doc/html/index.html

Pybind11
--------

* https://github.com/pybind/pybind11
* https://pybind11.readthedocs.io/en/stable/

Links
-----

* `Interfacing with C - Scipy lecture notes <https://www.scipy-lectures.org/advanced/interfacing_with_c/interfacing_with_c.html>`_

.. https://cppyy.readthedocs.io/en/latest/

* `SciPy Cookbook <https://scipy-cookbook.readthedocs.io/>`_
