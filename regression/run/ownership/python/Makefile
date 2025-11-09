# Copyright Shroud Project Developers. See LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
#
# #######################################################################
#
#  Compile the python ownership module
#
ifndef top
top = ../../..
endif

testdir = ownership
pymodule = ownership.so

AS_SHARED := TRUE
include $(top)/regression/run/defaults.mk

INCLUDE = \
    -I$(top)/regression/reference/ownership \
    -I$(top)/regression/run/ownership \
    $(PYTHON_INC)

VPATH = \
    $(top)/regression/reference/ownership \
    $(top)/regression/run/ownership \
    $(top)/regression/run/ownership/python

PY_OBJS = \
    ownership.o \
    pyClass1type.o \
    pyownershipmodule.o \
    pyownershiputil.o

all : ownership.so simple

ownership.so : $(PY_OBJS)
	$(CXX) $(LDFLAGS) -o $@ $^ $(LIBS)

simple : testpython.o $(PY_OBJS)
	$(CXX) -pthread -o $@ $^ $(PYTHON_LIB)
#	g++ -pthread -o $@ $^ $(PYTHON_LIB)

setup:
	$(PYTHON) setup.py build

