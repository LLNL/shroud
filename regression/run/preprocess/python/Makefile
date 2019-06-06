# Copyright (c) 2017-2019, Lawrence Livermore National Security, LLC. 
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
#  Compile the python preprocess module
#
include $(top)/regression/run/defaults.mk

INCLUDE = \
    -I$(top)/regression/reference/preprocess \
    -I$(top)/regression/run/preprocess \
    $(PYTHON_INC)

VPATH = \
    $(top)/regression/reference/preprocess \
    $(top)/regression/run/preprocess \
    $(top)/regression/run/preprocess/python

OBJS = \
    pypreprocessmodule.o \
    pypreprocesshelper.o \
    pyUser1type.o \
    pyUser2type.o

CXXFLAGS += $(SHARED)

all : preprocess.so

preprocess.so : $(OBJS)
	$(CXX) $(LD_SHARED) -o $@ $^ $(LIBS)

clean :
	rm -f *.so *.o
.PHONY : clean

print-debug:
	@echo PYTHON=$(PYTHON)
	@echo PYTHON_PREFIX=$(PYTHON_PREFIX)
	@echo PYTHON_VER=$(PYTHON_VER)

