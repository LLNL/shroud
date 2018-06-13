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
# targets to aid development
#


top := $(CURDIR)

PYTHONEXE := python2

PYTHON := $(shell which $(PYTHONEXE))
python.dir := $(dir $(PYTHON))
venv := $(dir $(PYTHON))virtualenv

include $(top)/tests/defaults.mk

LUA = $(shell which lua)

# build/temp-linux-x86_64-2.7
tempdir := build/temp.$(PLATFORM)-$(PYTHON_VER)
testsdir := $(top)/tests
venv.dir := $(top)/$(tempdir)/venv

# if virtualenv is created us it, else depend on python in path
ifneq ($(wildcard $(venv.dir)),)
python.dir := $(venv.dir)/bin
PYTHON := $(venv.dir)/bin/$(PYTHONEXE)
endif

export PYTHON PYTHONEXE
export LUA

compiler = gcc
export compiler

########################################################################
# For development:
# make virtualenv
# make develop

# Create a virtual environment.
# Include system site-packages to get numpy
virtualenv : $(venv.dir)
$(venv.dir) :
	$(venv) --system-site-packages $(venv.dir)

develop :
	$(PYTHON) setup.py develop

# python must have sphinx installed or else it reports
# error: invalid command 'build_sphinx'
docs :
	$(PYTHON) setup.py build_sphinx

test :
	$(PYTHON) setup.py test
#	$(PYTHON) -m unittest tests


requirements.txt :
	$(python.dir)/pip freeze > $@

# Pattern rule to make directories.
%/.. : ; $(at)test -d $(dir $@) || mkdir -p $(dir $@)

########################################################################
#
# Compile code in tutrial and string directory
# Used to make sure the generated wrappers work.
#
TESTDIRS = \
    $(tempdir)/run/tutorial/..\
    $(tempdir)/run/tutorial/python/.. \
    $(tempdir)/run/tutorial/lua/.. \
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

fortran : tutorial vectors strings clibrary ownership

# Compile the generated Fortran wrapper
tutorial vectors forward strings clibrary ownership : testdirs
	$(MAKE) \
	    -C $(tempdir)/run/$@ \
	    -f $(top)/tests/run/$@/Makefile \
	    top=$(top) $@

tutorial-c : testdirs
	$(MAKE) \
	    -C $(tempdir)/run/tutorial \
	    -f $(top)/tests/run/tutorial/Makefile \
	    top=$(top) testc

tutorial-cpp : testdirs
	$(MAKE) \
	    -C $(tempdir)/run/tutorial \
	    -f $(top)/tests/run/tutorial/Makefile \
	    top=$(top) maincpp

test-c : tutorial-c

# Run the Fortran tests
test-fortran : fortran
	$(tempdir)/run/tutorial/tutorial
	$(tempdir)/run/vectors/vectors
	$(tempdir)/run/strings/strings
	$(tempdir)/run/clibrary/clibrary
	$(tempdir)/run/ownership/ownership

# Compile the generated Python wrapper
py-tutorial : testdirs
	$(MAKE) \
	    -C $(tempdir)/run/tutorial/python \
	    -f $(top)/tests/run/tutorial/python/Makefile \
	    PYTHON=$(PYTHON) top=$(top) all

py-strings : testdirs
	$(MAKE) \
	    -C $(tempdir)/run/strings/python \
	    -f $(top)/tests/run/strings/python/Makefile \
	    PYTHON=$(PYTHON) top=$(top) all

py-clibrary : testdirs
	$(MAKE) \
	    -C $(tempdir)/run/clibrary/python \
	    -f $(top)/tests/run/clibrary/python/Makefile \
	    PYTHON=$(PYTHON) top=$(top) all

py-ownership : testdirs
	$(MAKE) \
	    -C $(tempdir)/run/ownership/python \
	    -f $(top)/tests/run/ownership/python/Makefile \
	    PYTHON=$(PYTHON) top=$(top) all

# Run the Python tests
test-python-tutorial : py-tutorial
	export PYTHONPATH=$(top)/$(tempdir)/run/tutorial/python; \
	$(PYTHON_BIN) $(top)/tests/run/tutorial/python/test.py

test-python-strings : py-strings
	export PYTHONPATH=$(top)/$(tempdir)/run/strings/python; \
	$(PYTHON_BIN) $(top)/tests/run/strings/python/test.py

test-python-clibrary : py-clibrary
	export PYTHONPATH=$(top)/$(tempdir)/run/clibrary/python; \
	$(PYTHON_BIN) $(top)/tests/run/clibrary/python/test.py

test-python-ownership : py-ownership
	export PYTHONPATH=$(top)/$(tempdir)/run/ownership/python; \
	$(PYTHON_BIN) $(top)/tests/run/ownership/python/test.py

test-python : test-python-tutorial test-python-strings test-python-clibrary test-python-ownership

# Compile the geneated Lua wrapper
lua-tutorial : testdirs
	$(MAKE) \
	    -C $(tempdir)/run/tutorial/lua \
	    -f $(top)/tests/run/tutorial/lua/Makefile \
	    LUA=$(LUA) top=$(top) all

# Run the Lua test
test-lua : lua-tutorial
#	export LUA_PATH=$(top)/$(tempdir)/run/tutorial/lua;
	cd $(top)/$(tempdir)/run/tutorial/lua; \
	$(LUA_BIN) $(top)/tests/run/tutorial/lua/test.lua

test-all : test-c test-fortran test-python test-lua

test-clean :
	rm -rf $(tempdir)/test
	rm -rf $(tempdir)/run

########################################################################
#
# Run the sample YAML files and compare output
# make do-test
# make do-test do-test-args=tutorial
#
do-test :
	@export TEST_OUTPUT_DIR=$(top)/$(tempdir)/test; \
	export TEST_INPUT_DIR=$(top)/tests; \
	export EXECUTABLE_DIR=$(python.dir); \
	$(PYTHON) tests/do_test.py $(do-test-args)

# replace test answers
do-test-replace :
	@export TEST_OUTPUT_DIR=$(top)/$(tempdir)/test; \
	export TEST_INPUT_DIR=$(top)/tests; \
	export EXECUTABLE_DIR=$(python.dir); \
	$(PYTHON) tests/do_test.py -r  $(do-test-args)

########################################################################

print-debug:
	@echo LUA=$(LUA)
	@echo PYTHON=$(PYTHON)
	@echo PYTHON_PREFIX=$(PYTHON_PREFIX)
	@echo PYTHON_VER=$(PYTHON_VER)
	@echo PLATFORM=$(PLATFORM)
	@echo tempdir=$(tempdir)

distclean:
	rm -rf build shroud.egg-info shroud/*.pyc tests/*.pyc
#	rm -rf dist
#	rm -rf .eggs

.PHONY : virtualenv develop docs test testdirs
.PHONY : fortran test-fortran tutorial vectors strings clibrary ownership
.PHONY : tutorial-c tutorial-cpp
.PHONY : test-python
.PHONY : py-tutorial test-python-tutorial
.PHONY : py-strings  test-python-strings
.PHONY : py-clibrary test-python-clibrary
.PHONY : py-ownership test-python-ownership
.PHONY : test-lua lua-tutorial
.PHONY : test-all test-clean
.PHONY : do-test do-test-replace print-debug
.PHONY : distclean
