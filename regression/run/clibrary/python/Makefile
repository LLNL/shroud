# Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and
# other Shroud Project Developers.
# See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
#
#  Compile the python clibrary module
#
include $(top)/regression/run/defaults.mk

INCLUDE = \
    -I$(top)/regression/reference/clibrary \
    -I$(top)/regression/run/clibrary \
    $(PYTHON_INC)

VPATH = \
    $(top)/regression/reference/clibrary \
    $(top)/regression/run/clibrary \
    $(top)/regression/run/clibrary/python

OBJS = \
    clibrary.o \
    pyClibrarymodule.o

LOCAL_CFLAGS += $(SHARED)

all : clibrary.so simple

clibrary.so : $(OBJS)
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

