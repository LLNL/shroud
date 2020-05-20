# Copyright (c) 2017-2020, Lawrence Livermore National Security, LLC and
# other Shroud Project Developers.
# See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
#
# #######################################################################
#
#  Compile the python cxxlibrary module
#
include $(top)/regression/run/defaults.mk

INCLUDE = \
    -I$(top)/regression/reference/cxxlibrary \
    -I$(top)/regression/run/cxxlibrary \
    $(PYTHON_INC)

VPATH = \
    $(top)/regression/reference/cxxlibrary \
    $(top)/regression/run/cxxlibrary

OBJS = \
    cxxlibrary.o \
    pycxxlibrarymodule.o

CXXFLAGS += $(SHARED)

all : cxxlibrary.so

cxxlibrary.so : $(OBJS)
	$(CXX) $(LD_SHARED) -o $@ $^ $(LIBS)

clean :
	rm -f *.so *.o
.PHONY : clean

print-debug:
	@echo PYTHON=$(PYTHON)
	@echo PYTHON_PREFIX=$(PYTHON_PREFIX)
	@echo PYTHON_VER=$(PYTHON_VER)

