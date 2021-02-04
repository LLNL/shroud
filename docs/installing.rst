.. Copyright (c) 2017-2021, Lawrence Livermore National Security, LLC and
   other Shroud Project Developers.
   See the top-level COPYRIGHT file for details.

   SPDX-License-Identifier: (BSD-3-Clause)

Installing
==========

The easiest way to install Shroud is via pip which will fetch a file from
`pypi <https://pypi.org>`_

.. code-block:: sh

    pip install llnl-shroud

This will install Shroud into the same directory as pip.
A virtual environment can be created if another destination directory
is desired.
For details see the
`python docs <https://packaging.python.org/tutorials/installing-packages/#creating-virtual-environments>`_
    
The source is available from `github.com/LLNL/shroud <https://github.com/LLNL/shroud>`_
A `shiv <https://github.com/linkedin/shiv>`_ packaged executable is also available
at `github releases <https://github.com/LLNL/shroud/releases>`_.
This is an executable file which contains Shroud and PyYAML and uses the Python3 in
the user's path.

**Shroud** is written in Python and has been tested with version 2.7 and 3.4+.
It requires the module:

  * PyYAML   https://github.com/yaml/pyyaml


After downloading the source:

.. code-block:: sh

    python setup.py install

This will create the script *shroud* in the same directory as Python.

Since shroud installs into Python's bin directory, it may be desirable to setup
a virtual environment to try it out:

.. code-block:: sh

    $ cd my_project_folder
    $ virtualenv my_project
    $ source my_project/bin/activate
    $ cd path/to/shroud/source
    $ python setup.py install

This will create an executable at ``my_project/bin/shroud``.
This version requires the virtual environment to run and 
may be difficult to share with others.

It's possible to create a standalone executable with
`shiv <https://github.com/linkedin/shiv>`_:

.. code-block:: sh

    $ cd path/to/shroud/source
    $shiv --python '/usr/bin/env python3' -c shroud -o dist/shroud.pyz .

A file *shroud.pyz* is created which bundles all of shroud and pyYAML into
a single file.  It uses the python on your path to run.

Building wrappers with CMake
----------------------------

Shroud can produce a CMake macro file with the option ``-cmake``. 
This option can be incorporated into a CMakefile as:

.. code-block:: cmake

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

The path to Shroud must be defined to CMake.  It can be defined on the command line as:

.. code-block:: sh

    cmake -DSHROUD_EXECUTABLE=/full/path/bin/shroud

The ``add_shroud`` macro can then be used in other ``CMakeLists.txt`` files as:

.. code-block:: cmake

    add_shroud(
        YAML_INPUT_FILE      ${YAML_INPUT_FILE}
        C_FORTRAN_OUTPUT_DIR c_fortran
    )

``CMake`` will treat all Fortran files as free format with the command:

.. code-block:: cmake

    set(CMAKE_Fortran_FORMAT FREE)


Building Python extensions
--------------------------

``setup.py`` can be used to build the extension module from the files created by shroud.
This example is drawn from the ``run/tutorial`` example.  You must provide the paths
to the input YAML file and the C++ library source files:

.. code-block:: python

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

The directory structure is layed out as:

.. code-block:: text

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
