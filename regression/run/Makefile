# Copyright (c) 2017-2021, Lawrence Livermore National Security, LLC and
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

# build/temp-
#             run/$(testdir)/prob
# relrunpath- relative directory, allow multiple compiler builds
testdir = .
relrunpath = $(tempdir)/run/$(testdir)

include $(top)/regression/run/defaults.mk

# ( directory, command )
# Save output to file and record PASS/FAIL
ifeq ($(LOGOUTPUT),1)
run-command = cd $1 && rm -f PASS FAIL ; { $2 >| out 2>&1; } && touch PASS || touch FAIL
else
run-command = cd $1 && $2
endif

######################################################################
# Compile a Fortran test
fortran-% : $(relrunpath)/%/.. phony_explicit
	$(MAKE) \
	    -C $(relrunpath)/$* \
	    -f $(top)/regression/run/$*/Makefile \
	    top=$(top) $*

# Run a Fortran test
test-fortran-% : fortran-%
	$(call run-command, $(relrunpath)/$*, ./$*)

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

# Futher Interoperability with C
# gfortran 9.1
# intel 16.0.3
test-cfi : \
  test-fortran-strings-cfi \
  test-fortran-generic-cfi

######################################################################
# Compile a C test
c-% : $(relrunpath)/%/.. phony_explicit
	$(MAKE) \
	    -C $(relrunpath)/$* \
	    -f $(top)/regression/run/$*/Makefile \
	    top=$(top) testc


# Run the C tests
test-c-% : c-%
	$(call run-command, $(relrunpath)/$*, ./testc)

# Compile C tests
test-c : \
  test-c-types \
  test-c-classes \
  test-c-enum-c \
  test-c-namespace \
  test-c-struct-cxx \
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
cpp-% : $(relrunpath)/%/.. phony_explicit
	$(MAKE) \
	    -C $(relrunpath)/$* \
	    -f $(top)/regression/run/$*/Makefile \
	    top=$(top) maincpp

# Run a C++ test
test-cpp-% : cpp-%
	$(call run-command, $(relrunpath)/$*, ./maincpp)

test-cpp : \
  cpp-tutorial \
  cpp-templates \
  test-cpp-scope

######################################################################
# Compile the generated Python wrapper
pymod-% : $(relrunpath)/%/python/.. phony_explicit
	$(MAKE) \
	    -C $(relrunpath)/$*/python \
	    -f $(top)/regression/run/$*/python/Makefile \
	    PYTHON=$(PYTHON) top=$(top) all

# Run the Python tests
test-python-% : pymod-%
	export PYTHONPATH=$(top)/$(relrunpath)/$*/python; \
	$(call run-command, $(relrunpath)/$*/python, \
	$(python.exe) $(top)/regression/run/$*/python/test.py)

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
  test-python-ccomplex \
  test-python-clibrary \
  test-python-cxxlibrary \
  test-python-ownership \
  test-python-templates

# test-python-templates

######################################################################
# Compile the generated Lua wrapper
compile-lua-% : $(relrunpath)/%/lua/.. phony_explicit
	$(MAKE) \
	    -C $(relrunpath)/$*/lua \
	    -f $(top)/regression/run/$*/lua/Makefile \
	    LUA=$(LUA) top=$(top) all

# Run the Lua test
test-lua-% : compile-lua-%
#	export LUA_PATH=$(top)/$(relrunpath)/tutorial/lua;
	$(call run-command, $(relrunpath)/$*/lua, \
	$(LUA_BIN) $(top)/regression/run/$*/lua/test.lua)

test-lua : \
    test-lua-tutorial \
    test-lua-classes

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
  test-c-struct-cxx \
  test-fortran-struct-c \
  test-fortran-struct-cxx \
  test-python-struct-numpy-c \
  test-python-struct-numpy-cxx \
  test-python-struct-class-c \
  test-python-struct-class-cxx

.PHONY : test-c
.PHONY : test-fortran
.PHONY : test-python
.PHONY : test-lua
.PHONY : test-all
.PHONY : test-struct

.PHONY: phony_explicit
phony_explicit: 

# Avoid deleting directories created to run tests.
# They are created via pattern rules.
.SECONDARY: \
    $(relrunpath)/ccomplex/.. \
    $(relrunpath)/ccomplex/python/.. \
    $(relrunpath)/cdesc/.. \
    $(relrunpath)/classes/.. \
    $(relrunpath)/classes/lua/.. \
    $(relrunpath)/classes/python/.. \
    $(relrunpath)/clibrary/.. \
    $(relrunpath)/clibrary/python/.. \
    $(relrunpath)/cxxlibrary/.. \
    $(relrunpath)/cxxlibrary/python/.. \
    $(relrunpath)/enum-c/.. \
    $(relrunpath)/enum-c/python/.. \
    $(relrunpath)/forward/.. \
    $(relrunpath)/generic/.. \
    $(relrunpath)/generic-cfi/.. \
    $(relrunpath)/namespace/.. \
    $(relrunpath)/namespace/python/.. \
    $(relrunpath)/ownership/.. \
    $(relrunpath)/ownership/python/.. \
    $(relrunpath)/pointers-c/..\
    $(relrunpath)/pointers-cxx/..\
    $(relrunpath)/pointers-numpy-cxx/python/.. \
    $(relrunpath)/pointers-list-cxx/python/.. \
    $(relrunpath)/pointers-numpy-c/python/.. \
    $(relrunpath)/pointers-list-c/python/.. \
    $(relrunpath)/preprocess/..\
    $(relrunpath)/preprocess/python/.. \
    $(relrunpath)/preprocess/.. \
    $(relrunpath)/arrayclass/.. \
    $(relrunpath)/arrayclass/python/.. \
    $(relrunpath)/scope/.. \
    $(relrunpath)/statement/.. \
    $(relrunpath)/strings/.. \
    $(relrunpath)/strings/python/.. \
    $(relrunpath)/strings-cfi/.. \
    $(relrunpath)/struct-c/.. \
    $(relrunpath)/struct-cxx/.. \
    $(relrunpath)/struct-numpy-c/python/.. \
    $(relrunpath)/struct-numpy-cxx/python/.. \
    $(relrunpath)/struct-class-c/python/.. \
    $(relrunpath)/struct-class-cxx/python/.. \
    $(relrunpath)/struct-list-c/python/.. \
    $(relrunpath)/structlist/python/.. \
    $(relrunpath)/struct-py-c/python/.. \
    $(relrunpath)/struct-py-cxx/python/.. \
    $(relrunpath)/templates/.. \
    $(relrunpath)/templates/python/.. \
    $(relrunpath)/tutorial/..\
    $(relrunpath)/tutorial/lua/.. \
    $(relrunpath)/tutorial/python/.. \
    $(relrunpath)/types/..\
    $(relrunpath)/types/python/..\
    $(relrunpath)/vectors/..\
    $(relrunpath)/vectors-list/python/.. \
    $(relrunpath)/vectors-numpy/python/..

test-clean-testdir :
	rm -rf $(relrunpath)
.PHONY : test-clean-testdir
