# Copyright (c) 2017-2021, Lawrence Livermore National Security, LLC and
# other Shroud Project Developers.
# See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
#
# #######################################################################
#
#  Compile the python arrayclass module
#
include $(top)/regression/run/defaults.mk

INCLUDE = \
    -I$(top)/regression/reference/arrayclass \
    -I$(top)/regression/run/arrayclass \
    $(PYTHON_INC)

VPATH = \
    $(top)/regression/reference/arrayclass \
    $(top)/regression/run/arrayclass

#    arrayclass.o 
OBJS = \
    pyarrayclassmodule.o \
    pyarrayclassutil.o \
    pyArrayWrappertype.o

CXXFLAGS += $(SHARED)

all : arrayclass.so

arrayclass.so : $(OBJS)
	$(CXX) $(LD_SHARED) -o $@ $^ $(LIBS)

clean :
	rm -f *.so *.o
.PHONY : clean

print-debug:
	@echo PYTHON=$(PYTHON)
	@echo PYTHON_PREFIX=$(PYTHON_PREFIX)
	@echo PYTHON_VER=$(PYTHON_VER)

