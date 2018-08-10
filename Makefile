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
#PYTHONEXE := python3

PYTHON := $(shell which $(PYTHONEXE))
python.dir := $(dir $(PYTHON))
venv := $(dir $(PYTHON))virtualenv
PYTHON_VER := $(shell $(PYTHON) -c "import sys;sys.stdout.write('{v[0]}.{v[1]}'.format(v=sys.version_info))")
PLATFORM := $(shell $(PYTHON) -c "import sys, sysconfig;sys.stdout.write(sysconfig.get_platform())")

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

include $(top)/regression/run/Makefile

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

setup-sqa :
#	$(PYTHON) -m pip install ruamel-yaml
#	$(PYTHON) -m pip install pyflakes
#	$(PYTHON) -m pip install mccabe
#	$(PYTHON) -m pip install flake8
#	$(PYTHON) -m pip install pylint
	$(python.dir)/pip install pylint

sqa :
#	$(python.dir)/pylint shroud/main.py
	pylint shroud/*.py > pylint.out


# Format code using black
# black requires Python3.6+
install-black :
	$(python.dir)/pip install black
black :
	LC_ALL=en_US.utf8 $(python.dir)/black shroud/*.py
#	LC_ALL=en_US.utf8 $(python.dir)/black regression/do-test.py
#	LC_ALL=en_US.utf8 $(python.dir)/black tests/*.py

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
	@export TEST_OUTPUT_DIR=$(top)/$(tempdir)/regression; \
	export TEST_INPUT_DIR=$(top)/regression; \
	export EXECUTABLE_DIR=$(python.dir); \
	$(PYTHON) regression/do-test.py $(do-test-args)

# replace test answers
do-test-replace :
	@export TEST_OUTPUT_DIR=$(top)/$(tempdir)/regression; \
	export TEST_INPUT_DIR=$(top)/regression; \
	export EXECUTABLE_DIR=$(python.dir); \
	$(PYTHON) regression/do-test.py -r $(do-test-args)

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
.PHONY : test-clean
.PHONY : do-test do-test-replace print-debug
.PHONY : distclean
