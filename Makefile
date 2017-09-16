# Copyright (c) 2017, Lawrence Livermore National Security, LLC. 
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

PYTHON := $(shell which python)
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
PYTHON := $(venv.dir)/bin/python
endif

export PYTHON
export LUA

########################################################################
# Create a virtual environment.
# Then 'make develop' will use the environement to install dependencies

virtualenv : $(venv.dir)
$(venv.dir) :
	$(venv) $(venv.dir)

develop :
	$(PYTHON) setup.py develop

# python must have sphinx installed or else it reports
# error: invalid command 'build_sphinx'
docs :
	$(PYTHON) setup.py build_sphinx

test :
	$(PYTHON) setup.py test
#	$(PYTHON) -m unittest tests


# Pattern rule to make directories.
%/.. : ; $(at)test -d $(dir $@) || mkdir -p $(dir $@)

########################################################################
#
# Compile code in tutrial and string directory
# Used to make sure the generated wrappers work.
#
TESTDIRS = \
    $(tempdir)/run-tutorial/..\
    $(tempdir)/run-tutorial/python/.. \
    $(tempdir)/run-tutorial/lua/.. \
    $(tempdir)/run-strings/..

testdirs : $(TESTDIRS)

fortran : tutorial strings

tutorial strings : testdirs
	$(MAKE) \
	    -C $(tempdir)/run-$@ \
	    -f $(top)/tests/run-$@/Makefile \
	    top=$(top) $@

test-fortran : fortran
	$(tempdir)/run-tutorial/tutorial
	$(tempdir)/run-strings/strings

py-tutorial : testdirs
	$(MAKE) \
	    -C $(tempdir)/run-tutorial/python \
	    -f $(top)/tests/run-tutorial/python/Makefile \
	    PYTHON=$(PYTHON) top=$(top) all

test-python : py-tutorial
	export PYTHONPATH=$(top)/$(tempdir)/run-tutorial/python; \
	$(PYTHON_BIN) $(top)/tests/run-tutorial/python/test.py	

lua-tutorial : testdirs
	$(MAKE) \
	    -C $(tempdir)/run-tutorial/lua \
	    -f $(top)/tests/run-tutorial/lua/Makefile \
	    LUA=$(LUA) top=$(top) all

test-lua : lua-tutorial
#	export LUA_PATH=$(top)/$(tempdir)/run-tutorial/lua;
	cd $(top)/$(tempdir)/run-tutorial/lua; \
	$(LUA_BIN) $(top)/tests/run-tutorial/lua/test.lua

test-all : test-fortran test-python test-lua

test-clean :
	rm -rf $(tempdir)

########################################################################
#
# Run the sample YAML files and compare output
#
do-test :
	@export TEST_OUTPUT_DIR=$(top)/$(tempdir)/test; \
	export TEST_INPUT_DIR=$(top)/tests; \
	export EXECUTABLE_DIR=$(python.dir); \
	$(PYTHON) tests/do_test.py

# replace test answers
do-test-replace :
	@export TEST_OUTPUT_DIR=$(top)/$(tempdir)/test; \
	export TEST_INPUT_DIR=$(top)/tests; \
	export EXECUTABLE_DIR=$(python.dir); \
	$(PYTHON) tests/do_test.py -r

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
.PHONY : fortran test-fortran tutorial strings
.PHONY : test-python py-tutorial
.PHONY : test-lua lua-tutorial
.PHONY : test-all test-clean
.PHONY : do-test do-test-replace print-debug
.PHONY : distclean
