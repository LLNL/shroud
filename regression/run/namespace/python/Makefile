# Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and
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
    namespace.o \
    pyouter_Cstruct1type.o \
    pynsmodule.o \
    pyns_outermodule.o \
    pynswork_ClassWorktype.o \
    pyns_nsworkmodule.o \
    pynsutil.o

LOCAL_CXXFLAGS += $(SHARED)

all : ns.so

ns.so : $(OBJS)
	$(CXX) $(LD_SHARED) -o $@ $^ $(LIBS)

namespace.o : namespace.hpp

clean :
	rm -f *.so *.o
.PHONY : clean

