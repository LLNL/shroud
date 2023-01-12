# Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and
# other Shroud Project Developers.
# See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
#
# #######################################################################
#
#  Compile the python classes module
#
include $(top)/regression/run/defaults.mk

INCLUDE = \
    -I$(top)/regression/reference/classes \
    -I$(top)/regression/run/classes \
    $(PYTHON_INC)

VPATH = \
    $(top)/regression/reference/classes \
    $(top)/regression/run/classes \
    $(top)/regression/run/classes/python

OBJS = \
    classes.o \
    pyclassesmodule.o \
    pyClass1type.o \
    pyClass2type.o \
    pySingletontype.o \
    pyShapetype.o \
    pyCircletype.o \
    pyclassesutil.o

LOCAL_CXXFLAGS += $(SHARED)

all : classes.so simple

classes.so : $(OBJS)
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

