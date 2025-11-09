# Copyright Shroud Project Developers. See LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
#
# #######################################################################
#
#  Compile the python typedefs module
#
testdir = typedefs-cxx
pymodule = typedefs.so

AS_SHARED := TRUE
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

PY_OBJS = \
    typedefs.o \
    pytypedefsmodule.o

all : typedefs.so #simple

typedefs.o : typedefs.cpp typedefs.h
	$(CXX) $(CXXFLAGS) $(INCLUDE) -c -o $@ $<

typedefs.so : $(PY_OBJS)
	$(CXX) $(LDFLAGS) -o $@ $^ $(LIBS)

simple : testpython.o $(PY_OBJS)
	$(CXX) -pthread -o $@ $^ $(PYTHON_LIB)
#	g++ -pthread -o $@ $^ $(PYTHON_LIB)
