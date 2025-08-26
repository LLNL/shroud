# Copyright Shroud Project Developers. See LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
# #######################################################################
#
#  Compile the python enum C++ module
#
testdir = enum-cxx
pymodule = cenum.so

AS_SHARED := TRUE
include $(top)/regression/run/defaults.mk

INCLUDE = \
    -I$(top)/regression/reference/enum-cxx \
    -I$(top)/regression/run/enum \
    $(PYTHON_INC)

VPATH = \
    $(top)/regression/reference/enum-cxx \
    ../../cxx \
    $(top)/regression/run/enum

PY_OBJS = \
    enum.o \
    pyenummodule.o

all : cenum.so

enum.o : enum.cpp enum.h
	$(CXX) $(CXXFLAGS) $(INCLUDE) -c -o $*.o $<

cenum.so : $(PY_OBJS)
	$(CXX) $(LDFLAGS) -o $@ $^ $(LIBS)

simple : testpython.o $(PY_OBJS)
	$(CXX) -pthread -o $@ $^ $(PYTHON_LIB)
#	g++ -pthread -o $@ $^ $(PYTHON_LIB)

pyenummodule.o : pyenummodule.hpp
