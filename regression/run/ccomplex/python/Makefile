# Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and
# other Shroud Project Developers.
# See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
#
#  Compile the python ccomplex module
#
include $(top)/regression/run/defaults.mk

INCLUDE = \
    -I$(top)/regression/reference/ccomplex \
    -I$(top)/regression/run/ccomplex \
    $(PYTHON_INC)

VPATH = \
    $(top)/regression/reference/ccomplex \
    $(top)/regression/run/ccomplex \
    $(top)/regression/run/ccomplex/python

OBJS = \
    ccomplex.o \
    pyccomplexmodule.o

LOCAL_CFLAGS += $(SHARED)

all : ccomplex.so simple

ccomplex.so : $(OBJS)
	$(CXX) $(LD_SHARED) -o $@ $^ $(LIBS)

simple : testpython.o $(OBJS)
	$(CXX) -pthread -o $@ $^ $(PYTHON_LIB)
#	g++ -pthread -o $@ $^ $(PYTHON_LIB)

clean :
	rm -f *.so *.o simple
.PHONY : clean

print-debug:
	@echo PYTHON=$(PYTHON)
	@echo PYTHON_PREFIX=$(PYTHON_PREFIX)
	@echo PYTHON_VER=$(PYTHON_VER)

