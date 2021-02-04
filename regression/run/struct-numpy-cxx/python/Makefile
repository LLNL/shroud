# Copyright (c) 2017-2021, Lawrence Livermore National Security, LLC and
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
    -I$(top)/regression/reference/struct-numpy-cxx \
    -I$(top)/regression/run/struct \
    $(PYTHON_INC)

VPATH = \
    $(top)/regression/reference/struct-numpy-cxx

OBJS = \
    struct.o \
    pystructmodule.o \
    pystructutil.o

CXXFLAGS += $(SHARED)

all : cstruct.so

# Convert C++ file to C
struct.cpp : $(top)/regression/run/struct/struct.c
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

