# Copyright Shroud Project Developers. See LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
#
# #######################################################################
#
#  Compile the python char module
#
testdir = char-cxx
pymodule = char.so

AS_SHARED := TRUE
include $(top)/regression/run/defaults.mk

INCLUDE = \
    -I$(top)/regression/reference/char-cxx \
    -I$(top)/regression/run \
    -I$(top)/regression/run/char \
    $(PYTHON_INC)

VPATH = \
    $(top)/regression/reference/char-cxx \
    $(top)/regression/run/char \
    ../../cxx

PY_OBJS = \
    char.o \
    pycharmodule.o

all : char.so

char.o : char.cpp char.h
	$(CXX) $(CXXFLAGS) $(INCLUDE) -c -o $*.o $<

char.so : $(PY_OBJS)
	$(CXX) $(LDFLAGS) -o $@ $^ $(LIBS)

simple : testpython.o $(PY_OBJS)
	$(CXX) -pthread -o $@ $^ $(PYTHON_LIB)
#	g++ -pthread -o $@ $^ $(PYTHON_LIB)
