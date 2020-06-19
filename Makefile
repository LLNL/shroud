# Copyright (c) 2017-2020, Lawrence Livermore National Security, LLC and
# other Shroud Project Developers.
# See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
#
########################################################################
#
# targets to aid development
#


top := $(CURDIR)

PYTHONEXE := python2
PYTHONEXE := python3

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

# If venv.dir is created then use it, else depend on python in path.
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

# For Python3 use venv module.  This solves the problem where virtualenv
# in the path does not match the python (like toss3).

# Create a virtual environment.
# Include system site-packages to get numpy
virtualenv : $(venv.dir)
$(venv.dir) :
	$(PYTHON) -m venv --system-site-packages $(venv.dir)
virtualenv2 :
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


# NOTE: black and flake8 set line length to 80
# Format code using black
# black requires Python3.6+
install-black :
	$(python.dir)/pip install black
black.opt = --line-length=80
black :
	LC_ALL=en_US.utf8 $(python.dir)/black $(black.opt) shroud/*.py
	LC_ALL=en_US.utf8 $(python.dir)/black $(black.opt) regression/do-test.py
	LC_ALL=en_US.utf8 $(python.dir)/black $(black.opt) tests/*.py

flake8 :
	flake8 shroud/*.py

# Sort import statements
install-isort :
	$(python.dir)/pip install isort
isort:
	isort shroud/*.py

install-pybindgen:
	$(python.dir)/pip install pybindgen

########################################################################
# Distributing at pypi
# make install-twine   (needs python3)

install-twine :
	$(python.dir)/pip install twine
sdist :
	$(python.dir)/python setup.py sdist bdist_wheel
twine-check:
	$(python.dir)/twine check dist/shroud-*.tar.gz
testpypi:
	$(python.dir)/twine upload -r testpypi dist/*
pypi:
	$(python.dir)/twine upload dist/*

.PHONY : install-twine sdist testpypi pypi

########################################################################
# Creating shiv executable
# This puts all of shroud into a single file.

install-shiv :
	$(python.dir)/pip install shiv

shiv-file :
	$(python.dir)/shiv --python '/usr/bin/env python3' -c shroud -o dist/shroud.pyz .

########################################################################
# python must have sphinx installed or else it reports
# error: invalid command 'build_sphinx'
docs :
	$(PYTHON) setup.py build_sphinx --builder html
#--build-dir build/sphinx/html
#/usr/bin/sphinx-build -b -E html source build\html
pdf :
	$(PYTHON) setup.py build_sphinx -b latex
	$(MAKE) -C build/sphinx/latex all-pdf

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
