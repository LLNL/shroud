# Copyright (c) 2017-2019, Lawrence Livermore National Security, LLC.
#
# Produced at the Lawrence Livermore National Laboratory
#
# LLNL-CODE-738041.
#
# All rights reserved.
#
# This file is part of Shroud.
#
# For details about use and distribution, please read LICENSE.
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
    $(tempdir)/run/types/..\
    $(tempdir)/run/pointers/..\
    $(tempdir)/run/pointers/python/.. \
    $(tempdir)/run/pointers-list/python/.. \
    $(tempdir)/run/pointers-numpy-c/python/.. \
    $(tempdir)/run/pointers-list-c/python/.. \
    $(tempdir)/run/struct/..\
    $(tempdir)/run/preprocess/..\
    $(tempdir)/run/preprocess/python/.. \
    $(tempdir)/run/vectors/..\
    $(tempdir)/run/vectors/python/.. \
    $(tempdir)/run/vectors/lua/.. \
    $(tempdir)/run/preprocess/.. \
    $(tempdir)/run/forward/.. \
    $(tempdir)/run/strings/.. \
    $(tempdir)/run/strings/python/.. \
    $(tempdir)/run/clibrary/.. \
    $(tempdir)/run/clibrary/python/.. \
    $(tempdir)/run/clibrary-list/python/.. \
    $(tempdir)/run/ownership/.. \
    $(tempdir)/run/ownership/python/.. \
    $(tempdir)/run/templates/..

testdirs : $(TESTDIRS)

######################################################################
# Compile a Fortran test
fortran-% : $(tempdir)/run/%/.. phony_explicit
	$(MAKE) \
	    -C $(tempdir)/run/$* \
	    -f $(top)/regression/run/$*/Makefile \
	    top=$(top) $*

# run a Fortran test
test-fortran-% : fortran-%
	$(tempdir)/run/$*/$*

# Run the Fortran tests
test-fortran : \
  test-fortran-tutorial \
  test-fortran-types \
  test-fortran-pointers \
  test-fortran-struct \
  test-fortran-vectors \
  test-fortran-preprocess \
  test-fortran-strings \
  test-fortran-clibrary \
  test-fortran-ownership \
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
  test-c-tutorial \
  test-c-templates

######################################################################
# Compile a C++ test
cpp-% : $(tempdir)/run/%/.. phony_explicit
	$(MAKE) \
	    -C $(tempdir)/run/$* \
	    -f $(top)/regression/run/$*/Makefile \
	    top=$(top) maincpp

test-cpp : \
  cpp-tutorial \
  cpp-templates

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
  test-python-strings \
  test-python-pointers \
  test-python-pointers-list \
  test-python-pointers-numpy-c \
  test-python-pointers-list-c \
  test-python-clibrary \
  test-python-ownership

######################################################################
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

######################################################################
test-all : test-c test-fortran test-python test-lua


.PHONY : test-c
.PHONY : test-python
.PHONY : test-lua lua-tutorial
.PHONY : test-all

.PHONY: phony_explicit
phony_explicit: 
