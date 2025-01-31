# Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
# other Shroud Project Developers.
# See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
# #######################################################################
#
#  Compile the python types module
#
include $(top)/regression/run/defaults.mk

INCLUDE = \
    -I$(top)/regression/reference/types \
    -I$(top)/regression/run/types \
    $(PYTHON_INC)

VPATH = \
    $(top)/regression/reference/types \
    $(top)/regression/run/types \
    $(top)/regression/run/types/python

OBJS = \
    types.o \
    pytypesmodule.o

LOCAL_CXXFLAGS += $(SHARED)

all : shtypes.so #simple

shtypes.so : $(OBJS)
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

pytypesmodule.o : pytypesmodule.hpp
pytypesutil.o : pytypesmodule.hpp
