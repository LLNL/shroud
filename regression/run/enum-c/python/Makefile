# Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and
# other Shroud Project Developers.
# See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
# #######################################################################
#
#  Compile the python enum module
#
include $(top)/regression/run/defaults.mk

INCLUDE = \
    -I$(top)/regression/reference/enum-c \
    -I$(top)/regression/run/enum-c \
    $(PYTHON_INC)

VPATH = \
    $(top)/regression/reference/enum-c \
    $(top)/regression/run/enum-c \
    $(top)/regression/run/enum-c/python

OBJS = \
    enum.o \
    pyenummodule.o

LOCAL_CFLAGS += $(SHARED)

all : cenum.so

cenum.so : $(OBJS)
	$(CXX) $(LD_SHARED) -o $@ $^ $(LIBS)

simple : testpython.o $(OBJS)
	$(CXX) -pthread -o $@ $^ $(PYTHON_LIB)
#	g++ -pthread -o $@ $^ $(PYTHON_LIB)

pyenummodule.o : pyenummodule.h

clean :
	rm -f *.so *.o simple
.PHONY : clean

print-debug:
	@echo PYTHON=$(PYTHON)
	@echo PYTHON_PREFIX=$(PYTHON_PREFIX)
	@echo PYTHON_VER=$(PYTHON_VER)

