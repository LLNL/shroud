# Copyright Shroud Project Developers. See LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
#
# #######################################################################
#
#  Compile the python struct module
#
testdir = struct-class-cxx
pymodule = cstruct.so

AS_SHARED := TRUE
include $(top)/regression/run/defaults.mk

INCLUDE = \
    -I$(top)/regression/reference/struct-class-cxx \
    -I$(top)/regression/run/struct \
    $(PYTHON_INC)

VPATH = \
    $(top)/regression/reference/struct-class-cxx \
    $(top)/regression/run/struct \
    ../../cxx

PY_OBJS = \
    struct.o \
    pystructmodule.o \
    pystructutil.o \
    pyCstruct1type.o \
    pyCstruct_ptrtype.o \
    pyCstruct_listtype.o \
    pyCstruct_numpytype.o \
    pyArrays1type.o

all : cstruct.so

cstruct.so : $(PY_OBJS)
	$(CXX) $(LDFLAGS) -o $@ $^ $(LIBS)

struct.o : struct.cpp struct.h
	$(CXX) $(CXXFLAGS) $(INCLUDE) -c -o $*.o $<


