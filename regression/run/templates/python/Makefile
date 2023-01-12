# Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and
# other Shroud Project Developers.
# See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
#
# #######################################################################
#
#  Compile the python templates module
#
include $(top)/regression/run/defaults.mk

INCLUDE = \
    -I$(top)/regression/reference/templates \
    -I$(top)/regression/run/templates \
    $(PYTHON_INC)

VPATH = \
    $(top)/regression/reference/templates \
    $(top)/regression/run/templates \
    $(top)/regression/run/templates/python

OBJS = \
    templates.o \
    pyinternal_ImplWorker1type.o \
    pyinternal_ImplWorker2type.o \
    pyWorkertype.o \
    pytemplatesutil.o \
    pytemplatesmodule.o \
    pytemplates_internalmodule.o \
    pytemplates_stdmodule.o \
    pyuser_inttype.o \
    pystd_vector_doubletype.o \
    pystd_vector_inttype.o


LOCAL_CXXFLAGS += $(SHARED)

all : templates.so simple

templates.so : $(OBJS)
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

