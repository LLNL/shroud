# Copyright Shroud Project Developers. See LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
#
# #######################################################################
#
#  Compile the python strings module
#
testdir = strings
pymodule = strings.so

AS_SHARED := TRUE
include $(top)/regression/run/defaults.mk

INCLUDE = \
    -I$(top)/regression/reference/strings \
    -I$(top)/regression/run \
    -I$(top)/regression/run/strings \
    $(PYTHON_INC)

VPATH = \
    $(top)/regression/reference/strings \
    $(top)/regression/run/strings \
    $(top)/regression/run/strings/python

PY_OBJS = \
    strings.o \
    pystringsmodule.o

all : strings.so simple

strings.so : $(PY_OBJS)
	$(CXX) $(LDFLAGS) -o $@ $^ $(LIBS)

simple : testpython.o $(PY_OBJS)
	$(CXX) -pthread -o $@ $^ $(PYTHON_LIB)
#	g++ -pthread -o $@ $^ $(PYTHON_LIB)

