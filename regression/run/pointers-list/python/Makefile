# Copyright (c) 2018-2019, Lawrence Livermore National Security, LLC. 
#
# Produced at the Lawrence Livermore National Laboratory 
#
# LLNL-CODE-738041.
#
# All rights reserved. 
#
# This file is part of Shroud.
#
# For details about use and distribution, please read LICENSE.
#
# #######################################################################
#
#  Compile the python pointers module
#
include $(top)/regression/run/defaults.mk

INCLUDE = \
    -I$(top)/regression/reference/pointers-list \
    -I$(top)/regression/run/pointers \
    $(PYTHON_INC)

VPATH = \
    $(top)/regression/reference/pointers-list \
    $(top)/regression/run/pointers \
    $(top)/regression/run/pointers-list/python

OBJS = \
    pointers.o \
    pypointersmodule.o \
    pypointersutil.o

CXXFLAGS += $(SHARED)

all : pointers.so
# simple

pointers.so : $(OBJS)
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

