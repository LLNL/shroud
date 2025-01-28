# Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and
# other Shroud Project Developers.
# See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
#
# #######################################################################
#
#  Compile the python char module
#
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

OBJS = \
    char.o \
    pycharmodule.o

LOCAL_CXXFLAGS += $(SHARED)

all : char.so

char.o : char.cpp char.h
	$(CXX) $(LOCAL_CXXFLAGS) $(INCLUDE) -c -o $*.o $<

char.so : $(OBJS)
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

