# Copyright (c) 2017-2021, Lawrence Livermore National Security, LLC and
# other Shroud Project Developers.
# See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
#
# #######################################################################
#
#  Compile the python vectors module
#
include $(top)/regression/run/defaults.mk

INCLUDE = \
    -I$(top)/regression/reference/vectors-list \
    -I$(top)/regression/run/vectors \
    $(PYTHON_INC)

VPATH = \
    $(top)/regression/reference/vectors-list \
    $(top)/regression/run/vectors \
    $(top)/regression/run/vectors-list/python

OBJS = \
    vectors.o \
    pyvectorsmodule.o

LOCAL_CXXFLAGS += $(SHARED)

all : vectors.so #simple

vectors.so : $(OBJS)
	$(CXX) $(LD_SHARED) -o $@ $^ $(LIBS)

simple : testpython.o $(OBJS)
	$(CXX) -pthread -o $@ $^ $(PYTHON_LIB)
#	g++ -pthread -o $@ $^ $(PYTHON_LIB)

clean :
	rm -f *.so *.o simple
.PHONY : clean

print-debug:
	@echo PYTHON=$(PYTHON)
	@echo PYTHON_PREFIX=$(PYTHON_PREFIX)
	@echo PYTHON_VER=$(PYTHON_VER)


vectors.o : vectors.hpp
pyvectorsmodule.o : pyvectorsmodule.hpp
pyvectorsutil.o : pyvectorsmodule.hpp
