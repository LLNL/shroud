# Copyright (c) 2017-2018, Lawrence Livermore National Security, LLC. 
# Produced at the Lawrence Livermore National Laboratory 
#
# LLNL-CODE-738041.
# All rights reserved. 
#
# This file is part of Shroud.  For details, see
# https://github.com/LLNL/shroud. Please also read shroud/LICENSE.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#
# * Redistributions of source code must retain the above copyright
#   notice, this list of conditions and the disclaimer below.
# 
# * Redistributions in binary form must reproduce the above copyright
#   notice, this list of conditions and the disclaimer (as noted below)
#   in the documentation and/or other materials provided with the
#   distribution.
#
# * Neither the name of the LLNS/LLNL nor the names of its contributors
#   may be used to endorse or promote products derived from this
#   software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL LAWRENCE
# LIVERMORE NATIONAL SECURITY, LLC, THE U.S. DEPARTMENT OF ENERGY OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
########################################################################
#
#  Compile and test generated wrappers.
#
compiler = gcc
export compiler

include $(top)/regression/run/defaults.mk

########################################################################
#
# Compile code in tutrial and string directory
# Used to make sure the generated wrappers work.
#
TESTDIRS = \
    $(tempdir)/run/tutorial/..\
    $(tempdir)/run/tutorial/python/.. \
    $(tempdir)/run/tutorial/lua/.. \
    $(tempdir)/run/templates/..\
    $(tempdir)/run/vectors/..\
    $(tempdir)/run/vectors/python/.. \
    $(tempdir)/run/vectors/lua/.. \
    $(tempdir)/run/forward/.. \
    $(tempdir)/run/strings/.. \
    $(tempdir)/run/strings/python/.. \
    $(tempdir)/run/clibrary/.. \
    $(tempdir)/run/clibrary/python/.. \
    $(tempdir)/run/ownership/.. \
    $(tempdir)/run/ownership/python/..

testdirs : $(TESTDIRS)

fortran : tutorial vectors strings clibrary ownership templates

# Compile the generated Fortran wrapper
tutorial vectors forward strings clibrary ownership templates : testdirs
	$(MAKE) \
	    -C $(tempdir)/run/$@ \
	    -f $(top)/regression/run/$@/Makefile \
	    top=$(top) $@

tutorial-c : testdirs
	$(MAKE) \
	    -C $(tempdir)/run/tutorial \
	    -f $(top)/regression/run/tutorial/Makefile \
	    top=$(top) testc

tutorial-cpp : testdirs
	$(MAKE) \
	    -C $(tempdir)/run/tutorial \
	    -f $(top)/regression/run/tutorial/Makefile \
	    top=$(top) maincpp

templates-c : testdirs
	$(MAKE) \
	    -C $(tempdir)/run/templates \
	    -f $(top)/regression/run/templates/Makefile \
	    top=$(top) testc

templates-cpp : testdirs
	$(MAKE) \
	    -C $(tempdir)/run/templates \
	    -f $(top)/regression/run/templates/Makefile \
	    top=$(top) maincpp

test-c : tutorial-c

# Run the Fortran tests
test-fortran : fortran
	$(tempdir)/run/tutorial/tutorial
	$(tempdir)/run/vectors/vectors
	$(tempdir)/run/strings/strings
	$(tempdir)/run/clibrary/clibrary
	$(tempdir)/run/ownership/ownership
	$(tempdir)/run/templates/templates

# Compile the generated Python wrapper
py-tutorial : testdirs
	$(MAKE) \
	    -C $(tempdir)/run/tutorial/python \
	    -f $(top)/regression/run/tutorial/python/Makefile \
	    PYTHON=$(PYTHON) top=$(top) all

py-strings : testdirs
	$(MAKE) \
	    -C $(tempdir)/run/strings/python \
	    -f $(top)/regression/run/strings/python/Makefile \
	    PYTHON=$(PYTHON) top=$(top) all

py-clibrary : testdirs
	$(MAKE) \
	    -C $(tempdir)/run/clibrary/python \
	    -f $(top)/regression/run/clibrary/python/Makefile \
	    PYTHON=$(PYTHON) top=$(top) all

py-ownership : testdirs
	$(MAKE) \
	    -C $(tempdir)/run/ownership/python \
	    -f $(top)/regression/run/ownership/python/Makefile \
	    PYTHON=$(PYTHON) top=$(top) all

# Run the Python tests
test-python-tutorial : py-tutorial
	export PYTHONPATH=$(top)/$(tempdir)/run/tutorial/python; \
	$(PYTHON_BIN) $(top)/regression/run/tutorial/python/test.py

test-python-strings : py-strings
	export PYTHONPATH=$(top)/$(tempdir)/run/strings/python; \
	$(PYTHON_BIN) $(top)/regression/run/strings/python/test.py

test-python-clibrary : py-clibrary
	export PYTHONPATH=$(top)/$(tempdir)/run/clibrary/python; \
	$(PYTHON_BIN) $(top)/regression/run/clibrary/python/test.py

test-python-ownership : py-ownership
	export PYTHONPATH=$(top)/$(tempdir)/run/ownership/python; \
	$(PYTHON_BIN) $(top)/regression/run/ownership/python/test.py

test-python : test-python-tutorial test-python-strings test-python-clibrary test-python-ownership

# Compile the geneated Lua wrapper
lua-tutorial : testdirs
	$(MAKE) \
	    -C $(tempdir)/run/tutorial/lua \
	    -f $(top)/regression/run/tutorial/lua/Makefile \
	    LUA=$(LUA) top=$(top) all

# Run the Lua test
test-lua : lua-tutorial
#	export LUA_PATH=$(top)/$(tempdir)/run/tutorial/lua;
	cd $(top)/$(tempdir)/run/tutorial/lua; \
	$(LUA_BIN) $(top)/regression/run/tutorial/lua/test.lua

test-all : test-c test-fortran test-python test-lua


.PHONY : fortran test-fortran tutorial vectors strings clibrary ownership
.PHONY : tutorial-c tutorial-cpp
.PHONY : test-python
.PHONY : py-tutorial test-python-tutorial
.PHONY : py-strings  test-python-strings
.PHONY : py-clibrary test-python-clibrary
.PHONY : py-ownership test-python-ownership
.PHONY : test-lua lua-tutorial
.PHONY : test-all
