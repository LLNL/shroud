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
#  Compile the python clibrary module
#
include $(top)/regression/run/defaults.mk

INCLUDE = \
    -I$(top)/regression/reference/clibrary \
    -I$(top)/regression/run/clibrary \
    $(PYTHON_INC)

VPATH = \
    $(top)/regression/reference/clibrary \
    $(top)/regression/run/clibrary \
    $(top)/regression/run/clibrary/python

OBJS = \
    clibrary.o \
    pyClibrarymodule.o \
    pyClibraryhelper.o

CFLAGS += $(SHARED)

all : clibrary.so simple

clibrary.so : $(OBJS)
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

