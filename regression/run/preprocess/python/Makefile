# Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
# other Shroud Project Developers.
# See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
#
# #######################################################################
#
#  Compile the python preprocess module
#
include $(top)/regression/run/defaults.mk

INCLUDE = \
    -I$(top)/regression/reference/preprocess \
    -I$(top)/regression/run/preprocess \
    $(PYTHON_INC)

VPATH = \
    $(top)/regression/reference/preprocess \
    $(top)/regression/run/preprocess \
    $(top)/regression/run/preprocess/python

OBJS = \
    pypreprocessmodule.o \
    pypreprocessutil.o \
    pyUser1type.o \
    pyUser2type.o

LOCAL_CXXFLAGS += $(SHARED)

all : preprocess.so

preprocess.so : $(OBJS)
	$(CXX) $(LD_SHARED) -o $@ $^ $(LIBS)

clean :
	rm -f *.so *.o
.PHONY : clean

print-debug:
	@echo PYTHON=$(PYTHON)
	@echo PYTHON_PREFIX=$(PYTHON_PREFIX)
	@echo PYTHON_VER=$(PYTHON_VER)

