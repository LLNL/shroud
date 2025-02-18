# Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
# other Shroud Project Developers.
# See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
#
# #######################################################################
#
#  Compile the python pointers module
#
testdir = pointers-numpy-cxx
pymodule = pointers.so

AS_SHARED := TRUE
include $(top)/regression/run/defaults.mk

INCLUDE = \
    -I$(top)/regression/reference/pointers-numpy-cxx \
    -I$(top)/regression/run/pointers \
    $(PYTHON_INC)

VPATH = \
    $(top)/regression/reference/pointers-numpy-cxx \
    $(top)/regression/run/pointers/python \
    $(top)/regression/run/pointers \
    ../../cxx

PY_OBJS = \
    pointers.o \
    pypointersmodule.o

all : pointers.so simple

pointers.o : pointers.cpp pointers.h
	$(CXX) $(CXXFLAGS) $(INCLUDE) -c -o $*.o $<

pointers.so : $(PY_OBJS)
	$(CXX) $(LDFLAGS) $^ -o $@ $(LIBS)

simple : testpython.o $(PY_OBJS)
	$(CXX) -pthread -o $@ $^ $(PYTHON_LIB)
#	g++ -pthread -o $@ $^ $(PYTHON_LIB)

