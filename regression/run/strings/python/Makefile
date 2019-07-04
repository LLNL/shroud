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
#  Compile the python strings module
#
include $(top)/regression/run/defaults.mk

INCLUDE = \
    -I$(top)/regression/reference/strings \
    -I$(top)/regression/run/strings \
    $(PYTHON_INC)

VPATH = \
    $(top)/regression/reference/strings \
    $(top)/regression/run/strings \
    $(top)/regression/run/strings/python

OBJS = \
    strings.o \
    pystringsmodule.o \
    pystringsutil.o

CXXFLAGS += $(SHARED)

all : strings.so simple

strings.so : $(OBJS)
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

