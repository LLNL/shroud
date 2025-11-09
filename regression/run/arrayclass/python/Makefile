# Copyright Shroud Project Developers. See LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
#
# #######################################################################
#
#  Compile the python arrayclass module
#
testdir = arrayclass
pymodule = arrayclass.so

AS_SHARED := TRUE
include $(top)/regression/run/defaults.mk

INCLUDE = \
    -I$(top)/regression/reference/arrayclass \
    -I$(top)/regression/run/arrayclass \
    $(PYTHON_INC)

VPATH = \
    $(top)/regression/reference/arrayclass \
    $(top)/regression/run/arrayclass

#    arrayclass.o 
PY_OBJS = \
    pyarrayclassmodule.o \
    pyarrayclassutil.o \
    pyArrayWrappertype.o

all : arrayclass.so

arrayclass.so : $(PY_OBJS)
	$(CXX) $(LDFLAGS) -o $@ $^ $(LIBS)


