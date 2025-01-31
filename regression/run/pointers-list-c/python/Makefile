# Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
# other Shroud Project Developers.
# See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
#
# #######################################################################
#
#  Compile the python pointers module
#  regression/run/pointers-list-c/python
#
include $(top)/regression/run/defaults.mk

INCLUDE = \
    -I$(top)/regression/reference/pointers-list-c \
    -I$(top)/regression/run/pointers \
    $(PYTHON_INC)

VPATH = \
    $(top)/regression/reference/pointers-list-c \
    $(top)/regression/run/pointers

OBJS = \
    pointers.o \
    pypointersmodule.o

LOCAL_CFLAGS += $(SHARED)

all : pointers.so

pointers.so : $(OBJS)
	$(CC) $(LD_SHARED) -o $@ $^ $(LIBS)

simple : testpython.o $(OBJS)
	$(CC) -pthread -o $@ $^ $(PYTHON_LIB)
#	g++ -pthread -o $@ $^ $(PYTHON_LIB)

clean :
	rm -f *.so *.o simple
.PHONY : clean

print-debug:
	@echo PYTHON=$(PYTHON)
	@echo PYTHON_PREFIX=$(PYTHON_PREFIX)
	@echo PYTHON_VER=$(PYTHON_VER)

