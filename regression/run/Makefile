# Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and
# other Shroud Project Developers.
# See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
#
########################################################################
#
#  Compile and test generated wrappers.
#
# Initialization
# 3.80 for order-only dependency.
need := 3.80
ok := $(filter $(need),$(firstword $(sort $(MAKE_VERSION) $(need))))
$(if $(ok),,$(error version $(need) of gmake or greater is required))

compiler = gcc
export compiler

# build/temp-
#             run/$(testdir)/prob
# relrunpath- relative directory, allow multiple compiler builds
testdir = .
relrunpath = $(tempdir)/run/$(testdir)

include $(top)/regression/run/defaults.mk

# ( prefix, directory, command )
# Save output to file and record PASS/FAIL
ifeq ($(LOGOUTPUT),1)
run-command = cd $2 && rm -f $1.PASS $1.FAIL ; { $3 >| $1.out 2>&1; } && touch $1.PASS || ( touch $1.FAIL; false )
else
run-command = cd $2 && $3
endif

######################################################################
#
# Several tests compile the test code with both C and C++.
# Copy the .c suffix to a .cpp suffix file to compile as C++.
# Done as an order-only dependency since the referrence file will not
# change during the compilation test.
#

.PHONY : sync-cxx
sync-cxx : \
  $(relrunpath)/cxx/char.cpp \
  $(relrunpath)/cxx/enum.cpp \
  $(relrunpath)/cxx/funptr.cpp \
  $(relrunpath)/cxx/pointers.cpp \
  $(relrunpath)/cxx/struct.cpp \
  $(relrunpath)/cxx/typedefs.cpp

$(relrunpath)/cxx/char.cpp : regression/run/char/char.c | $(relrunpath)/cxx/..
	cp $< $@
$(relrunpath)/cxx/enum.cpp : regression/run/enum/enum.c | $(relrunpath)/cxx/..
	cp $< $@
$(relrunpath)/cxx/funptr.cpp : regression/run/funptr/funptr.c | $(relrunpath)/cxx/..
	cp $< $@
$(relrunpath)/cxx/struct.cpp : regression/run/struct/struct.c | $(relrunpath)/cxx/..
	cp $< $@
$(relrunpath)/cxx/pointers.cpp : regression/run/pointers/pointers.c | $(relrunpath)/cxx/..
	cp $< $@
$(relrunpath)/cxx/typedefs.cpp : regression/run/typedefs/typedefs.c | $(relrunpath)/cxx/..
	cp $< $@


######################################################################
# Fortran

fortran-test-list-std = \
  test-fortran-tutorial \
  test-fortran-types \
  test-fortran-typemap32 \
  test-fortran-typemap64 \
  test-fortran-classes \
  test-fortran-forward \
  test-fortran-enum-c \
  test-fortran-enum-cxx \
  test-fortran-typedefs-c \
  test-fortran-typedefs-cxx \
  test-fortran-namespace \
  test-fortran-pointers-c \
  test-fortran-pointers-cxx \
  test-fortran-pointers-c-f \
  test-fortran-pointers-cxx-f \
  test-fortran-arrayclass \
  test-fortran-struct-c \
  test-fortran-struct-cxx \
  test-fortran-funptr-c \
  test-fortran-funptr-cxx \
  test-fortran-vectors \
  test-fortran-cdesc \
  test-fortran-preprocess \
  test-fortran-char-cxx \
  test-fortran-strings \
  test-fortran-ccomplex \
  test-fortran-clibrary \
  test-fortran-cxxlibrary \
  test-fortran-defaultarg32 \
  test-fortran-defaultarg64 \
  test-fortran-ownership \
  test-fortran-generic \
  test-fortran-statement \
  test-fortran-templates \
  test-fortran-shared \
  test-fortran-wrap-cxx

fortran-test-list-cfi = \
  test-fortran-pointers-cfi \
  test-fortran-strings-cfi \
  test-fortran-generic-cfi

fortran-test-list = $(fortran-test-list-std) $(fortran-test-list-cfi)

# Compile a Fortran test
fortran-% : | $(relrunpath)/%/.. sync-cxx
	$(call run-command,build-f,$(relrunpath)/$*,\
	$(MAKE) -f $(top)/regression/run/$*/Makefile top=$(top) $*)

# Run a Fortran test
.PHONY : $(fortran-test-list)
$(fortran-test-list) : test-fortran-% : fortran-% | $(relrunpath)/%/..
	$(call run-command,test-f,$(relrunpath)/$*,./$*)

# Run the Fortran tests
.PHONY : test-fortran
test-fortran : $(fortran-test-list-std)

# Futher Interoperability with C
# gfortran 9.1
# intel 16.0.3
.PHONY : test-cfi
test-cfi : $(fortran-test-list-cfi)


######################################################################
# C

c-test-list = \
  test-c-types \
  test-c-typemap32 \
  test-c-typemap64 \
  test-c-classes \
  test-c-enum-cxx \
  test-c-typedefs-cxx \
  test-c-namespace \
  test-c-pointers-cxx \
  test-c-pointers-cxx-c \
  test-c-struct-cxx \
  test-c-funptr-cxx \
  test-c-vectors \
  test-c-strings \
  test-c-statement \
  test-c-templates

# Compile a C test
c-% : | $(relrunpath)/%/.. sync-cxx
	$(call run-command,build-c,$(relrunpath)/$*,\
	$(MAKE) -f $(top)/regression/run/$*/Makefile top=$(top) testc)

# Run the C tests
.PHONY : $(c-test-list)
$(c-test-list) : test-c-% : c-% | $(relrunpath)/%/..
	$(call run-command,test-c,$(relrunpath)/$*,./testc)

# Compile C tests
.PHONY : test-c
test-c : $(c-test-list)

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
# C++

cpp-test-list = \
  test-cpp-classes \
  test-cpp-templates \
  test-cpp-sgroup \
  test-cpp-shared \
  test-cpp-defaultarg32 \
  test-cpp-defaultarg64 \
  test-cpp-scope

# Compile a C++ test
cpp-% : | $(relrunpath)/%/..
	$(call run-command,build-cxx,$(relrunpath)/$*,\
	$(MAKE) -f $(top)/regression/run/$*/Makefile top=$(top) maincpp)

# Run a C++ test
.PHONY : $(cpp-test-list)
$(cpp-test-list) : test-cpp-% : cpp-% | $(relrunpath)/%/..
	$(call run-command,test-cxx,$(relrunpath)/$*,./maincpp)

.PHONY : test-cpp
test-cpp : $(cpp-test-list)

######################################################################
# Python
#  test-python-typedefs-c \
#  test-python-typedefs-cxx

python-test-list = \
  test-python-tutorial \
  test-python-types \
  test-python-classes \
  test-python-enum-c \
  test-python-enum-cxx \
  test-python-namespace \
  test-python-char-cxx \
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
  test-python-vectors-list \
  test-python-ccomplex \
  test-python-clibrary \
  test-python-cxxlibrary \
  test-python-ownership \
  test-python-templates

# Compile the generated Python wrapper
pymod-% : | $(relrunpath)/%/python/.. sync-cxx
	$(call run-command,build-py,$(relrunpath)/$*/python,\
	$(MAKE) -f $(top)/regression/run/$*/python/Makefile \
	    PYTHON=$(PYTHON) top=$(top) all)

# Run the Python tests
.PHONY : $(python-test-list)
$(python-test-list) : test-python-% : pymod-% | $(relrunpath)/%/python/..
	export PYTHONPATH=$(top)/$(relrunpath)/$*/python; \
	$(call run-command,test-py,$(relrunpath)/$*/python, \
	$(python.exe) $(top)/regression/run/$*/python/test.py)

.PHONY : test-python
test-python : $(python-test-list)

# test-python-templates

######################################################################
# Lua

lua-test-list = \
    test-lua-tutorial \
    test-lua-classes

# Compile the generated Lua wrapper
compile-lua-% : | $(relrunpath)/%/lua/..
	$(call run-command,build-lua,$(relrunpath)/$*/lua,\
	$(MAKE) -f $(top)/regression/run/$*/lua/Makefile \
	    LUA=$(LUA) top=$(top) all)

# Run the Lua test
.PHONY : $(lua-test-list)
$(lua-test-list) : test-lua-% : compile-lua-% | $(relrunpath)/%/lua/..
#	export LUA_PATH=$(top)/$(relrunpath)/tutorial/lua;
	$(call run-command,test-lua,$(relrunpath)/$*/lua, \
	$(LUA_BIN) $(top)/regression/run/$*/lua/test.lua)

.PHONY : test-lua
test-lua : $(lua-test-list)

######################################################################
.PHONY : test-all test-all-fortran-c
test-all : test-c test-fortran test-python test-lua

test-all-fortran-c : test-c test-fortran

# test specific yaml files
.PHONY : test-pointers
test-pointers : \
  test-fortran-pointers-c \
  test-fortran-pointers-cxx \
  test-python-pointers-numpy-c \
  test-python-pointers-numpy-cxx \
  test-python-pointers-list-c \
  test-python-pointers-list-cxx

.PHONY : test-struct
test-struct : \
  test-c-struct-cxx \
  test-fortran-struct-c \
  test-fortran-struct-cxx \
  test-python-struct-numpy-c \
  test-python-struct-numpy-cxx \
  test-python-struct-class-c \
  test-python-struct-class-cxx

.PHONY : test-typemap
test-typemap : \
  test-c-typemap32 \
  test-c-typemap64 \
  test-fortran-typemap32 \
  test-fortran-typemap64 \

test-clean-testdir :
	rm -rf $(relrunpath)
.PHONY : test-clean-testdir
