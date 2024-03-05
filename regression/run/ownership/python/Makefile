# Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and
# other Shroud Project Developers.
# See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
#
# #######################################################################
#
#  Compile the python ownership module
#
ifndef top
top = ../../..
endif

include $(top)/regression/run/defaults.mk

INCLUDE = \
    -I$(top)/regression/reference/ownership \
    -I$(top)/regression/run/ownership \
    $(PYTHON_INC)

VPATH = \
    $(top)/regression/reference/ownership \
    $(top)/regression/run/ownership \
    $(top)/regression/run/ownership/python

OBJS = \
    ownership.o \
    pyClass1type.o \
    pyownershipmodule.o \
    pyownershiputil.o

LOCAL_CXXFLAGS += $(SHARED)

all : ownership.so simple

ownership.so : $(OBJS)
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

setup:
	$(PYTHON) setup.py build

