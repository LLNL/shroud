# Copyright (c) 2017-2020, Lawrence Livermore National Security, LLC and
# other Shroud Project Developers.
# See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
#
########################################################################
#
#  Compile and test generated wrappers.
#
compiler = gcc
export compiler

include $(top)/regression/run/defaults.mk

######################################################################
# Compile a Fortran test
fortran-% : $(tempdir)/run/%/.. phony_explicit
	$(MAKE) \
	    -C $(tempdir)/run/$* \
	    -f $(top)/regression/run/$*/Makefile \
	    top=$(top) $*

# Run a Fortran test
test-fortran-% : fortran-%
	$(tempdir)/run/$*/$*

# Run the Fortran tests
test-fortran : \
  test-fortran-tutorial \
  test-fortran-types \
  test-fortran-classes \
  test-fortran-forward \
  test-fortran-enum-c \
  test-fortran-namespace \
  test-fortran-pointers-c \
  test-fortran-pointers-cxx \
  test-fortran-arrayclass \
  test-fortran-struct-c \
  test-fortran-struct-cxx \
  test-fortran-vectors \
  test-fortran-cdesc \
  test-fortran-preprocess \
  test-fortran-strings \
  test-fortran-ccomplex \
  test-fortran-clibrary \
  test-fortran-cxxlibrary \
  test-fortran-ownership \
  test-fortran-generic \
  test-fortran-statement \
  test-fortran-templates

######################################################################
# Compile a C test
c-% : $(tempdir)/run/%/.. phony_explicit
	$(MAKE) \
	    -C $(tempdir)/run/$* \
	    -f $(top)/regression/run/$*/Makefile \
	    top=$(top) testc


# Run the C tests
test-c-% : c-%
	$(tempdir)/run/$*/testc

# Compile C tests
test-c : \
  test-c-types \
  test-c-classes \
  test-c-enum-c \
  test-c-namespace \
  test-c-statement \
  test-c-templates


##  test-c-tutorial 
##  test-c-types 
##  test-c-namespace 
#  test-c-pointers 
#  test-c-struct-c 
#  test-c-struct-cxx 
#  test-c-vectors 
#  test-c-preprocess 
#  test-c-strings 
#  test-c-clibrary 
#  test-c-ownership 
#  test-c-generic 
##  test-c-templates


######################################################################
# Compile a C++ test
cpp-% : $(tempdir)/run/%/.. phony_explicit
	$(MAKE) \
	    -C $(tempdir)/run/$* \
	    -f $(top)/regression/run/$*/Makefile \
	    top=$(top) maincpp

# Run a C++ test
test-cpp-% : cpp-%
	$(tempdir)/run/$*/maincpp

test-cpp : \
  cpp-tutorial \
  cpp-templates \
  test-cpp-scope

######################################################################
# Compile the generated Python wrapper
pymod-% : $(tempdir)/run/%/python/.. phony_explicit
	$(MAKE) \
	    -C $(tempdir)/run/$(subst pymod-,,$@)/python \
	    -f $(top)/regression/run/$(subst pymod-,,$@)/python/Makefile \
	    PYTHON=$(PYTHON) top=$(top) all

# Run the Python tests
test-python-% : pymod-%
	export PYTHONPATH=$(top)/$(tempdir)/run/$*/python; \
	$(python.exe) $(top)/regression/run/$*/python/test.py

test-python : \
  test-python-tutorial \
  test-python-types \
  test-python-classes \
  test-python-enum-c \
  test-python-namespace \
  test-python-strings \
  test-python-pointers-numpy-cxx \
  test-python-pointers-list-cxx \
  test-python-pointers-numpy-c \
  test-python-pointers-list-c \
  test-python-arrayclass \
  test-python-struct-numpy-c \
  test-python-struct-numpy-cxx \
  test-python-struct-class-c \
  test-python-struct-class-cxx \
  test-python-structlist \
  test-python-struct-py-c \
  test-python-struct-py-cxx \
  test-python-vectors-numpy \
  test-python-clibrary \
  test-python-cxxlibrary \
  test-python-ownership \
  test-python-templates

# test-python-templates

######################################################################
# Compile the geneated Lua wrapper
lua-tutorial : $(tempdir)/run/tutorial/lua/.. phony_explicit
	$(MAKE) \
	    -C $(tempdir)/run/tutorial/lua \
	    -f $(top)/regression/run/tutorial/lua/Makefile \
	    LUA=$(LUA) top=$(top) all

# Run the Lua test
test-lua : lua-tutorial
#	export LUA_PATH=$(top)/$(tempdir)/run/tutorial/lua;
	cd $(top)/$(tempdir)/run/tutorial/lua; \
	$(LUA_BIN) $(top)/regression/run/tutorial/lua/test.lua

######################################################################
test-all : test-c test-fortran test-python test-lua

# test specific yaml files
test-pointers : \
  test-fortran-pointers-c \
  test-fortran-pointers-cxx \
  test-python-pointers-numpy-c \
  test-python-pointers-numpy-cxx \
  test-python-pointers-list-c \
  test-python-pointers-list-cxx

test-struct : \
  test-fortran-struct-c \
  test-fortran-struct-cxx \
  test-python-struct-numpy-c \
  test-python-struct-numpy-cxx \
  test-python-struct-class-c \
  test-python-struct-class-cxx

.PHONY : test-c
.PHONY : test-python
.PHONY : test-lua lua-tutorial
.PHONY : test-all
.PHONY : test-struct

.PHONY: phony_explicit
phony_explicit: 

# Avoid deleting directories created to run tests.
# They are created via pattern rules.
.SECONDARY: \
    $(tempdir)/run/ccomplex/.. \
    $(tempdir)/run/ccomplex/python/.. \
    $(tempdir)/run/cdesc/.. \
    $(tempdir)/run/classes/.. \
    $(tempdir)/run/classes/python/.. \
    $(tempdir)/run/clibrary/.. \
    $(tempdir)/run/clibrary/python/.. \
    $(tempdir)/run/cxxlibrary/.. \
    $(tempdir)/run/cxxlibrary/python/.. \
    $(tempdir)/run/enum-c/.. \
    $(tempdir)/run/enum-c/python/.. \
    $(tempdir)/run/forward/.. \
    $(tempdir)/run/generic/.. \
    $(tempdir)/run/namespace/.. \
    $(tempdir)/run/namespace/python/.. \
    $(tempdir)/run/ownership/.. \
    $(tempdir)/run/ownership/python/.. \
    $(tempdir)/run/pointers-c/..\
    $(tempdir)/run/pointers-cxx/..\
    $(tempdir)/run/pointers-numpy-cxx/python/.. \
    $(tempdir)/run/pointers-list-cxx/python/.. \
    $(tempdir)/run/pointers-numpy-c/python/.. \
    $(tempdir)/run/pointers-list-c/python/.. \
    $(tempdir)/run/preprocess/..\
    $(tempdir)/run/preprocess/python/.. \
    $(tempdir)/run/preprocess/.. \
    $(tempdir)/run/arrayclass/.. \
    $(tempdir)/run/arrayclass/python/.. \
    $(tempdir)/run/scope/.. \
    $(tempdir)/run/statement/.. \
    $(tempdir)/run/strings/.. \
    $(tempdir)/run/strings/python/.. \
    $(tempdir)/run/struct-c/.. \
    $(tempdir)/run/struct-cxx/.. \
    $(tempdir)/run/struct-numpy-c/python/.. \
    $(tempdir)/run/struct-numpy-cxx/python/.. \
    $(tempdir)/run/struct-class-c/python/.. \
    $(tempdir)/run/struct-class-cxx/python/.. \
    $(tempdir)/run/struct-list-c/python/.. \
    $(tempdir)/run/structlist/python/.. \
    $(tempdir)/run/struct-py-c/python/.. \
    $(tempdir)/run/struct-py-cxx/python/.. \
    $(tempdir)/run/templates/.. \
    $(tempdir)/run/templates/python/.. \
    $(tempdir)/run/tutorial/..\
    $(tempdir)/run/tutorial/python/.. \
    $(tempdir)/run/tutorial/lua/.. \
    $(tempdir)/run/types/..\
    $(tempdir)/run/types/python/..\
    $(tempdir)/run/vectors/..\
    $(tempdir)/run/vectors-list/python/.. \
    $(tempdir)/run/vectors-numpy/python/..

