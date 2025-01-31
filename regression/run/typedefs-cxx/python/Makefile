# Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
# other Shroud Project Developers.
# See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
#
# #######################################################################
#
#  Compile the python typedefs module
#
include $(top)/regression/run/defaults.mk

INCLUDE = \
    -I$(top)/regression/reference/typedefs-cxx \
    -I$(top)/regression/run/typedefs \
    $(PYTHON_INC)

VPATH = \
    $(top)/regression/reference/typedefs-cxx \
    ../../cxx \
    $(top)/regression/run/typedefs \
    $(top)/regression/run/typedefs/python

OBJS = \
    typedefs.o \
    pytypedefsmodule.o

LOCAL_CXXFLAGS += $(SHARED)

all : typedefs.so #simple

typedefs.o : typedefs.cpp typedefs.h
	$(CXX) $(LOCAL_CXXFLAGS) $(TEST_CXXFLAGS) $(INCLUDE) -c -o $@ $<

typedefs.so : $(OBJS)
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

