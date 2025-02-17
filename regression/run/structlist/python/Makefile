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
testdir = structlist
pymodule = cstruct.so

AS_SHARED := TRUE
include $(top)/regression/run/defaults.mk

INCLUDE = \
    -I$(top)/regression/reference/structlist \
    -I$(top)/regression/run/struct \
    $(PYTHON_INC)

VPATH = \
    $(top)/regression/reference/structlist \
    $(top)/regression/run/struct

PY_OBJS = \
    struct.o \
    pystructmodule.o \
    pystructutil.o \
    pyArrays1type.o

all : cstruct.so

cstruct.so : $(PY_OBJS)
	$(CC) $(LDFLAGS) -o $@ $^ $(LIBS)


