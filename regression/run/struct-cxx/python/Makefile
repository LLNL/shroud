# Copyright (c) 2017-2020, Lawrence Livermore National Security, LLC and
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
    -I$(top)/regression/reference/struct-cxx \
    -I$(top)/regression/run/struct-c \
    $(PYTHON_INC)

VPATH = \
    $(top)/regression/reference/struct-cxx \
    $(top)/regression/run/struct-cxx/python

OBJS = \
    struct.o \
    pystructmodule.o \
    pystructutil.o \
    pyCstruct1type.o

CXXFLAGS += $(SHARED)

all : cstruct.so

# Convert C++ file to C
struct.cpp : $(top)/regression/run/struct-c/struct.c
	cp $^ $@

cstruct.so : $(OBJS)
	$(CXX) $(LD_SHARED) -o $@ $^ $(LIBS)

struct.o : struct.cpp

clean :
	rm -f *.so *.o
.PHONY : clean

print-debug:
	@echo PYTHON=$(PYTHON)
	@echo PYTHON_PREFIX=$(PYTHON_PREFIX)
	@echo PYTHON_VER=$(PYTHON_VER)

