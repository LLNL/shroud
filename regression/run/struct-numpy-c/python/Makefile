# Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
# other Shroud Project Developers.
# See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
#
# #######################################################################
#
#  Compile the python struct module
#
testdir = struct-numpy-c
pymodule = cstruct.so

AS_SHARED := TRUE
include $(top)/regression/run/defaults.mk

INCLUDE = \
    -I$(top)/regression/reference/struct-numpy-c \
    -I$(top)/regression/run/struct \
    $(PYTHON_INC)

VPATH = \
    $(top)/regression/reference/struct-numpy-c \
    $(top)/regression/run/struct

PY_OBJS = \
    struct.o \
    pystructmodule.o \
    pystructutil.o

all : cstruct.so

cstruct.so : $(PY_OBJS)
	$(CC) $(LDFLAGS) -o $@ $^ $(LIBS)
