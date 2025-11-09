# Copyright Shroud Project Developers. See LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
#
# #######################################################################
#
#  Compile the python pointers module
#  regression/run/pointers-numpy-c/python
#
testdir = pointers-numpy-c
pymodule = pointers.so

AS_SHARED := TRUE
include $(top)/regression/run/defaults.mk

INCLUDE = \
    -I$(top)/regression/reference/pointers-numpy-c \
    -I$(top)/regression/run/pointers \
    $(PYTHON_INC)

VPATH = \
    $(top)/regression/reference/pointers-numpy-c \
    $(top)/regression/run/pointers

PY_OBJS = \
    pointers.o \
    pypointersmodule.o

all : pointers.so

pointers.so : $(PY_OBJS)
	$(CC) $(LDFLAGS) $^ -o $@ $(LIBS)

simple : testpython.o $(PY_OBJS)
	$(CC) -pthread -o $@ $^ $(PYTHON_LIB)
#	g++ -pthread -o $@ $^ $(PYTHON_LIB)

