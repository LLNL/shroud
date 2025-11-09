# Copyright Shroud Project Developers. See LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
#
# #######################################################################
#
#  Compile the python templates module
#
testdir = templates
pymodule = templates.so

AS_SHARED := TRUE
include $(top)/regression/run/defaults.mk

INCLUDE = \
    -I$(top)/regression/reference/templates \
    -I$(top)/regression/run/templates \
    $(PYTHON_INC)

VPATH = \
    $(top)/regression/reference/templates \
    $(top)/regression/run/templates \
    $(top)/regression/run/templates/python

PY_OBJS = \
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

all : templates.so simple

templates.so : $(PY_OBJS)
	$(CXX) $(LDFLAGS) -o $@ $^ $(LIBS)

simple : testpython.o $(PY_OBJS)
	$(CXX) -pthread -o $@ $^ $(PYTHON_LIB)
#	g++ -pthread -o $@ $^ $(PYTHON_LIB)

