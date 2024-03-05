# Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and
# other Shroud Project Developers.
# See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
#
# #######################################################################
#
#  Compile the python struct module
#
include $(top)/regression/run/defaults.mk

INCLUDE = \
    -I$(top)/regression/reference/struct-py-cxx \
    -I$(top)/regression/run/struct \
    $(PYTHON_INC)

VPATH = \
    $(top)/regression/reference/struct-py-cxx \
    $(top)/regression/run/struct \
    ../../cxx

OBJS = \
    struct.o \
    pystructmodule.o \
    pyCstruct_as_classtype.o \
    pystructutil.o

LOCAL_CXXFLAGS += $(SHARED)

all : cstruct.so

cstruct.so : $(OBJS)
	$(CXX) $(LD_SHARED) -o $@ $^ $(LIBS)

struct.o : struct.cpp struct.h
	$(CXX) $(LOCAL_CXXFLAGS) $(INCLUDE) -c -o $*.o $<

clean :
	rm -f *.so *.o
.PHONY : clean

print-debug:
	@echo PYTHON=$(PYTHON)
	@echo PYTHON_PREFIX=$(PYTHON_PREFIX)
	@echo PYTHON_VER=$(PYTHON_VER)

