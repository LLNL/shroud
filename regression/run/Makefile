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
    $(tempdir)/run/templates/..\
    $(tempdir)/run/pointers/..\
    $(tempdir)/run/pointers/python/.. \
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

fortran : tutorial types pointers vectors strings clibrary ownership templates

# Compile the Fortran tests
tutorial types pointers vectors forward strings clibrary ownership templates : testdirs
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

# Compile C tests
clang : tutorial-c templates-c

# Run the Fortran tests
test-fortran : fortran
	$(tempdir)/run/tutorial/tutorial
	$(tempdir)/run/types/types
	$(tempdir)/run/pointers/pointers
	$(tempdir)/run/vectors/vectors
	$(tempdir)/run/strings/strings
	$(tempdir)/run/clibrary/clibrary
	$(tempdir)/run/ownership/ownership
	$(tempdir)/run/templates/templates

# run a Fortran test
test-fortran-% : %
	$(tempdir)/run/$(subst test-fortran-,,$@)/$(subst test-fortran-,,$@)
#	$(tempdir)/run/$(patsubst fortran-%,%,$@)/$(patsubst fortran-%,%,$@)

# Run a single Fortran test
# make ftest name=templates
ftest :
	$(tempdir)/run/$(name)/$(name)

# Run the C tests
test-c : clang
	$(tempdir)/run/tutorial/testc
	$(tempdir)/run/templates/testc

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

py-pointers : testdirs
	$(MAKE) \
	    -C $(tempdir)/run/pointers/python \
	    -f $(top)/regression/run/pointers/python/Makefile \
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
	$(python.exe) $(top)/regression/run/tutorial/python/test.py

test-python-strings : py-strings
	export PYTHONPATH=$(top)/$(tempdir)/run/strings/python; \
	$(python.exe) $(top)/regression/run/strings/python/test.py

test-python-pointers : py-pointers
	export PYTHONPATH=$(top)/$(tempdir)/run/pointers/python; \
	$(python.exe) $(top)/regression/run/pointers/python/test.py

test-python-clibrary : py-clibrary
	export PYTHONPATH=$(top)/$(tempdir)/run/clibrary/python; \
	$(python.exe) $(top)/regression/run/clibrary/python/test.py

test-python-ownership : py-ownership
	export PYTHONPATH=$(top)/$(tempdir)/run/ownership/python; \
	$(python.exe) $(top)/regression/run/ownership/python/test.py

test-python : test-python-tutorial test-python-strings test-python-pointers test-python-clibrary test-python-ownership

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


.PHONY : fortran test-fortran tutorial pointers vectors strings clibrary ownership templates
.PHONY : test-c
.PHONY : tutorial-c tutorial-cpp
.PHONY : templates-c
.PHONY : test-python
.PHONY : py-tutorial test-python-tutorial
.PHONY : py-strings  test-python-strings
.PHONY : py-pointers  test-python-pointers
.PHONY : py-clibrary test-python-clibrary
.PHONY : py-ownership test-python-ownership
.PHONY : test-lua lua-tutorial
.PHONY : test-all
