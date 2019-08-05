# Copyright (c) 2017-2019, Lawrence Livermore National Security, LLC and
# other Shroud Project Developers.
# See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

include $(top)/regression/run/defaults.mk

INCLUDE = \
    -I$(top)/regression/reference/namespace \
    -I$(top)/regression/run/namespace \
    $(PYTHON_INC)

VPATH = \
    $(top)/regression/reference/namespace \
    $(top)/regression/run/namespace \
    $(top)/regression/run/namespace/python

OBJS = \
    pyCstruct1type.o \
    pynsmodule.o \
    pyns_outermodule.o \
    pyns_stdmodule.o \
    pynsutil.o

CXXFLAGS += $(SHARED)

all : namespace.so

namespace.so : $(OBJS)
	$(CXX) $(LD_SHARED) -o $@ $^ $(LIBS)

clean :
	rm -f *.so *.o
.PHONY : clean

