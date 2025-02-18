# Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
# other Shroud Project Developers.
# See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
#
# #######################################################################
#
#  Compile the python typedefs module
#
testdir = typedefs-c
pymodule = typedefs.so

AS_SHARED := TRUE
include $(top)/regression/run/defaults.mk

INCLUDE = \
    -I$(top)/regression/reference/typedefs-c \
    -I$(top)/regression/run/typedefs \
    $(PYTHON_INC)

VPATH = \
    $(top)/regression/reference/typedefs-c \
    $(top)/regression/run/typedefs \
    $(top)/regression/run/typedefs/python

PY_OBJS = \
    typedefs.o \
    pytypedefsmodule.o

all : typedefs.so #simple

typedefs.so : $(PY_OBJS)
	$(CC) $(LDFLAGS) -o $@ $^ $(LIBS)

simple : testpython.o $(PY_OBJS)
	$(CC) -pthread -o $@ $^ $(PYTHON_LIB)
#	g++ -pthread -o $@ $^ $(PYTHON_LIB)
