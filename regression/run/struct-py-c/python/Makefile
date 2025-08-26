# Copyright Shroud Project Developers. See LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
#
# #######################################################################
#
#  Compile the python struct module
#
testdir = struct-py-c
pymodule = cstruct.so

AS_SHARED := TRUE
include $(top)/regression/run/defaults.mk

INCLUDE = \
    -I$(top)/regression/reference/struct-py-c \
    -I$(top)/regression/run/struct \
    $(PYTHON_INC)

VPATH = \
    $(top)/regression/reference/struct-py-c \
    $(top)/regression/run/struct

PY_OBJS = \
    struct.o \
    pystructmodule.o \
    pyCstruct_as_classtype.o \
    pystructutil.o

all : cstruct.so

cstruct.so : $(PY_OBJS)
	$(CC) $(LDFLAGS) -o $@ $^ $(LIBS)
