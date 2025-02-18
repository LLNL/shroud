# Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
# other Shroud Project Developers.
# See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
#
# #######################################################################
#
#  Compile the python struct module
#
testdir = struct-py-cxx
pymodule = cstruct.so

AS_SHARED := TRUE
include $(top)/regression/run/defaults.mk

INCLUDE = \
    -I$(top)/regression/reference/struct-py-cxx \
    -I$(top)/regression/run/struct \
    $(PYTHON_INC)

VPATH = \
    $(top)/regression/reference/struct-py-cxx \
    $(top)/regression/run/struct \
    ../../cxx

PY_OBJS = \
    struct.o \
    pystructmodule.o \
    pyCstruct_as_classtype.o \
    pystructutil.o

all : cstruct.so

cstruct.so : $(PY_OBJS)
	$(CXX) $(LDFLAGS) -o $@ $^ $(LIBS)

struct.o : struct.cpp struct.h
	$(CXX) $(CXXFLAGS) $(INCLUDE) -c -o $*.o $<
