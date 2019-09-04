# Copyright (c) 2017-2019, Lawrence Livermore National Security, LLC and
# other Shroud Project Developers.
# See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
#
# #######################################################################
#
#  Compile the python struct module
#
include $(top)/regression/run/defaults.mk

INCLUDE = \
    -I$(top)/regression/reference/struct-c \
    -I$(top)/regression/run/struct-c \
    $(PYTHON_INC)

VPATH = \
    $(top)/regression/reference/struct-c \
    $(top)/regression/run/struct-c \
    $(top)/regression/run/struct-c/python

OBJS = \
    struct.o \
    pystructmodule.o \
    pystructutil.o

CFLAGS += $(SHARED)

all : cstruct.so

cstruct.so : $(OBJS)
	$(CC) $(LD_SHARED) -o $@ $^ $(LIBS)

clean :
	rm -f *.so *.o
.PHONY : clean

print-debug:
	@echo PYTHON=$(PYTHON)
	@echo PYTHON_PREFIX=$(PYTHON_PREFIX)
	@echo PYTHON_VER=$(PYTHON_VER)

