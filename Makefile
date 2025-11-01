# Copyright Shroud Project Developers. See LICENSE file for details.
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
PYTHON := $(venv.dir)/bin/$(notdir $(PYTHONEXE))
endif

export PYTHON PYTHONEXE
export LUA

include $(top)/regression/run/Makefile


info:
	@echo PYTHON     = $(PYTHON)
	@echo PYTHON_VER = $(PYTHON_VER)
	@echo PLATFORM   = $(PLATFORM)

########################################################################
# For development:
# module load python/2.7.18
# module load python/3.10.8
# module load python/3.12.2

# make virtualenv
# make develop
# module load gcc/6.1.0   or newer

# For Python3 use venv module.  This solves the problem where virtualenv
# in the path does not match the python (like toss3).

# Create a virtual environment.
# Include system site-packages to get numpy
virtualenv : $(venv.dir)
$(venv.dir) :
	$(PYTHON) -m venv --system-site-packages $(venv.dir)
	$(venv.dir)/bin/pip install --upgrade pip
#wheel setuptools
virtualenv2 :
	$(venv) --system-site-packages $(venv.dir)

pipinstall :
	$(venv.dir)/bin/pip install .

develop :
	$(PYTHON) setup.py develop
#	$(PYTHON) setup.py egg_info --egg-base $(venv.dir) develop
#	$(venv.dir)/bin/pip install --editable .

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
# Creating pex executable
# This puts all of shroud into a single file.
# https://github.com/pantsbuild/pex

install-pex :
	$(python.dir)/pip install pex

# Use version in output file name.
pex-file : vernum = $(shell grep __version__ shroud/metadata.py | awk -F '"' '{print $$2}')
pex-file : dist-pex/..
	$(python.dir)/pex . -r requirements.txt --python-shebang="/usr/bin/env python3" \
          -e shroud.main:main -o dist-pex/shroud-$(vernum).pex
	cd dist-pex && ln --force --symbolic shroud-$(vernum).pex shroud.pex

# Test pex created executable
do-test-pex :
	@export TEST_OUTPUT_DIR=$(top)/$(tempdir)/regression; \
	export TEST_INPUT_DIR=$(top)/regression; \
	export EXECUTABLE_DIR=$(top)/dist-pex/shroud.pex; \
	$(PYTHON) regression/do-test.py $(do-test-args)

########################################################################
# Creating shiv executable
# This puts all of shroud into a single file.
# https://github.com/linkedin/shiv
# Note: Python 3.6+

install-shiv :
	$(python.dir)/pip install shiv

# Use version in output file name.
shiv-file : vernum = $(shell grep __version__ shroud/metadata.py | awk -F '"' '{print $$2}')
shiv-file : dist-shiv/..
	$(python.dir)/shiv --python '/usr/bin/env python3' -c shroud \
          -o dist-shiv/shroud-$(vernum).pyz .
	cd dist-shiv && ln --force --symbolic shroud-$(vernum).pyz shroud.pyz

# Test shiv created executable
do-test-shiv :
	@export TEST_OUTPUT_DIR=$(top)/$(tempdir)/regression; \
	export TEST_INPUT_DIR=$(top)/regression; \
	export EXECUTABLE_DIR=$(top)/dist-shiv/shroud.pyz; \
	$(PYTHON) regression/do-test.py $(do-test-args)

########################################################################
# Nuitka is a Python compiler written in Python.
# http://nuitka.net/
# hinted-compilation https://github.com/Nuitka/NUITKA-Utilities
#
# Create an executable at $(nuitka-root)/$(vernum)/shroud.dist/shroud

nuitka-root = dist-nuitka

install-nuitka :
	$(python.dir)/pip install nuitka

# $(python.dir)/python -m nuitka 
nuitka-options = $(python.dir)/nuitka3
nuitka-options += --standalone
nuitka-options += --follow-imports
#nuitka-options += --show-progress
#nuitka-options += --show-scons
#nuitka-options += --generate-c-only
nuitka-options += --remove-output
#nuitka-options += --output-dir=nuitka-work

# Use version in output file name.
nuitka-file : vernum = $(shell grep __version__ shroud/metadata.py | awk -F '"' '{print $$2}')
nuitka-file :
	CC=gcc $(nuitka-options) --output-dir=$(nuitka-root)/$(vernum) dist-nuitka/shroud.py

# Test nuitka created executable
do-test-nuitka : vernum = $(shell grep __version__ shroud/metadata.py | awk -F '"' '{print $$2}')
do-test-nuitka :
	@export TEST_OUTPUT_DIR=$(top)/$(tempdir)/regression; \
	export TEST_INPUT_DIR=$(top)/regression; \
	export EXECUTABLE_DIR=$(nuitka-root)/$(vernum)/shroud.dist/shroud; \
	$(PYTHON) regression/do-test.py $(do-test-args)

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
	$(PYTHON) -m unittest tests


requirements.txt :
	$(python.dir)/pip freeze > $@

# Pattern rule to make directories.
%/.. : ; $(at)test -d $(dir $@) || mkdir -p $(dir $@)

test-clean :
	rm -rf $(tempdir)/test
	rm -rf $(tempdir)/run

########################################################################
#
# Compare output of check_decl.py with previous run.
#
decl_file = check_decl.output
decl_path = $(tempdir)/$(decl_file)
decl_ref = $(testsdir)/$(decl_file)

.PHONY : test-decl-work
test-decl-work :
	rm -f $(decl_path) && \
	$(PYTHON) $(testsdir)/check_decl.py > $(decl_path) 2>&1

test-decl : test-decl-work
	diff $(decl_ref) $(decl_path)

test-decl-replace : test-decl-work
	cp $(decl_path) $(decl_ref)

test-decl-diff :
	tkdiff $(decl_ref) $(decl_path)

########################################################################
#
# Run the sample YAML files and compare output
# make do-test
# make do-test do-test-args=tutorial
#
do-test :
	@export TEST_OUTPUT_DIR=$(top)/$(tempdir)/regression; \
	export TEST_INPUT_DIR=$(top)/regression; \
	export EXECUTABLE_DIR=$(python.dir)/shroud; \
	$(PYTHON) regression/do-test.py $(do-test-args)

# replace test answers
do-test-replace :
	@export TEST_OUTPUT_DIR=$(top)/$(tempdir)/regression; \
	export TEST_INPUT_DIR=$(top)/regression; \
	export EXECUTABLE_DIR=$(python.dir)/shroud; \
	$(PYTHON) regression/do-test.py -r $(do-test-args)

########################################################################
# Run tests prior to a commit

test-commit :
	@$(MAKE) test-clean
	@$(MAKE) test
	@$(MAKE) test-decl
	@$(MAKE) do-test
#	@$(MAKE) test-all

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

.PHONY : virtualenv pipinstall develop docs pdf test testdirs
.PHONY : virtualenv2
.PHONY : test-clean
.PHONY : do-test do-test-replace print-debug
.PHONY : distclean

########################################################################

# ANSI color codes
none    := \033[0m
red     := \033[0;31m
green   := \033[0;32m
yellow  := \033[0;33m
blue    := \033[0;34m
magenta := \033[0;35m
cyan    := \033[0;36m
all_colors := none red green yellow blue magenta cyan
export $(all_colors) all_colors

# Shell command to unset the exported colors, when not on terminal.
setcolors = { [ -t 1 ] || unset $${all_colors}; }

# Macro cprint to be used in rules.
# Example: $(call cprint,"$${red}warning: %s$${none}\n" "a is not defined")
cprint = $(setcolors) && printf $(1)

# Macro cprint2 to be used in rules generated with $(eval ), i.e. expanded twice
# $(1) - Text to print, $(2) - color-name (optional)
cprint2 = $(setcolors) && \
  printf "$(if $(2),$$$${$(2)},$$$${green})%s$$$${none}\n" '$(1)'

.PHONY: printvars print-%
# Print the value of a variable named "foo".
# Usage: make print-foo
print-%:
	@$(call cprint,"%s is $${green}%s$${none} ($${cyan}%s$${none})\
	  (from $${magenta}%s$${none})\n" '$*' '$($*)' '$(value $*)'\
	  '$(origin $*)')

# Print the value of (nearly) all the variables.
# Usage: make printvars
printvars:
	@:;$(foreach V,$(sort $(.VARIABLES)),\
	$(if $(filter-out environ% default automatic,$(origin $V)),\
	$(info $(V)=$($V) ($(value $(V))))))
	@:

