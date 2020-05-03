# Copyright (c) 2017-2020, Lawrence Livermore National Security, LLC and
# other Shroud Project Developers.
# See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
#
# #######################################################################
#
#  Compile the python references module
#
include $(top)/regression/run/defaults.mk

INCLUDE = \
    -I$(top)/regression/reference/references \
    -I$(top)/regression/run/references \
    $(PYTHON_INC)

VPATH = \
    $(top)/regression/reference/references \
    $(top)/regression/run/references

#    references.o 
OBJS = \
    pyreferencesmodule.o \
    pyreferencesutil.o \
    pyArrayWrappertype.o

CXXFLAGS += $(SHARED)

all : references.so

references.so : $(OBJS)
	$(CC) $(LD_SHARED) -o $@ $^ $(LIBS)

clean :
	rm -f *.so *.o
.PHONY : clean

print-debug:
	@echo PYTHON=$(PYTHON)
	@echo PYTHON_PREFIX=$(PYTHON_PREFIX)
	@echo PYTHON_VER=$(PYTHON_VER)

