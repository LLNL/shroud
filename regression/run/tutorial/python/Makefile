# Copyright (c) 2017-2021, Lawrence Livermore National Security, LLC and
# other Shroud Project Developers.
# See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
#
# #######################################################################
#
#  Compile the python tutorial module
#
include $(top)/regression/run/defaults.mk

INCLUDE = \
    -I$(top)/regression/reference/tutorial \
    -I$(top)/regression/run/tutorial \
    $(PYTHON_INC)

VPATH = \
    $(top)/regression/reference/tutorial \
    $(top)/regression/run/tutorial \
    $(top)/regression/run/tutorial/python

OBJS = \
    tutorial.o \
    pyTutorialmodule.o

CXXFLAGS += $(SHARED)

all : tutorial.so simple

tutorial.so : $(OBJS)
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

