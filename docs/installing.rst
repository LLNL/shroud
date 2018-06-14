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

Installing
==========

The source is available from `github.com/LLNL/shroud <https://github.com/LLNL/shroud>`_

**Shroud** is written in Python and has been tested with version 2.7 and 3.4+.
It requires the module:

  * PyYAML https://pypi.python.org/pypi/PyYAML/3.11


After downloading the source::

    python setup.py install

This will create the script *shroud* in the same directory as Python.

Since shroud installs into Python's bin directory, it may be desirable to setup
a virtual environment to try it out::

   $ cd my_project_folder
   $ virtualenv my_project
   $ source my_project/bin/activate
   $ cd path/to/shroud/source
   $ python setup.py install

This will create an executable at ``my_project/bin/shroud``.
This version requires the virtual environment to run and 
may be difficult to share with others.

It's possible to create a standalone executable with
`pex <https://github.com/pantsbuild/pex>`_::

	$(pex.root)/bin/pex -f $(dist)
	  --python-shebang=$(python.root)/bin/python
	  shroud==$(version) -e shroud.main:main
	  -o shroud

*pex.root* is the path to the pex executable, *dist* is a directory
where the shroud distribution file is ( ``shroud.0.9.9.tar.gz`` ),
*python.root* is the path to the path executable, and version is
shroud version ( ``0.9.0`` ).  This will create a single file, shroud,
which can be installed anywhere on the system.  It will even work on
other machines, assuming *python.root* is the same.  Or leaving off
the ``--python-shebang`` option, it will use the version of Python in
your path.

Building wrappers with CMake
----------------------------

Shroud can produce a CMake macro file with the option ``-cmake``. 
This option can be incorporated into a CMakefile as::

    if(EXISTS ${SHROUD_EXECUTABLE})
        execute_process(COMMAND ${SHROUD_EXECUTABLE}
                        --cmake ${CMAKE_CURRENT_BINARY_DIR}/SetupShroud.cmake
                        ERROR_VARIABLE SHROUD_cmake_error
                        OUTPUT_STRIP_TRAILING_WHITESPACE )
        if(${SHROUD_cmake_error})
           message(FATAL_ERROR "Error from Shroud: ${SHROUD_cmake_error}")
        endif()
        include(${CMAKE_CURRENT_BINARY_DIR}/SetupShroud.cmake)
    endif()

The path to Shroud must be defined to CMake.  It can be defined on the command line as::

    cmake -DSHROUD_EXECUTABLE=/full/path/bin/shroud

The ``add_shroud`` macro can then be used in other ``CMakeLists.txt`` files as::

    add_shroud(
        YAML_INPUT_FILE      ${YAML_INPUT_FILE}
        C_FORTRAN_OUTPUT_DIR c_fortran
    )

``CMake`` will treat all Fortran files as free format with the command::

    set(CMAKE_Fortran_FORMAT FREE)


Building Python extensions
--------------------------

``setup.py`` can be used to build the extension module from the files created by shroud.
This example is drawn from the ``run/tutorial`` example.  You must provide the paths
to the input YAML file and the C++ library source files::

    import os
    from distutils.core import setup, Extension
    import shroud
    import numpy
    
    outdir = 'build/source'
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    config = shroud.create_wrapper('../../../tutorial.yaml',
                                   path=['../../..'],
                                   outdir=outdir)
    
    tutorial = Extension(
        'tutorial',
        sources = config.pyfiles + ['../tutorial.cpp'],
        include_dirs=[numpy.get_include(), '..']
    )
    
    setup(
        name='tutorial',
        version="0.0",
        description='shroud tutorial',
        author='xxx',
        author_email='yyy@zz',
        ext_modules=[tutorial],
    )

The directory structure is layed out as::

     tutorial.yaml
     run
       tutorial
         tutorial.cpp   # C++ library to wrap
         tutorial.hpp
         python
           setup.py     # setup file shown above
           build
              source
                # create by shroud
                pyClass1type.cpp
                pySingletontype.cpp
                pyTutorialmodule.cpp
                pyTutorialmodule.hpp
                pyTutorialhelper.cpp
              lib
                 tutorial.so   # generated module
