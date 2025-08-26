# Copyright Shroud Project Developers. See LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

testdir = namespace
pymodule = ns.so

AS_SHARED := TRUE
include $(top)/regression/run/defaults.mk

INCLUDE = \
    -I$(top)/regression/reference/namespace \
    -I$(top)/regression/run/namespace \
    $(PYTHON_INC)

VPATH = \
    $(top)/regression/reference/namespace \
    $(top)/regression/run/namespace \
    $(top)/regression/run/namespace/python

PY_OBJS = \
    namespace.o \
    pyouter_Cstruct1type.o \
    pynsmodule.o \
    pyns_outermodule.o \
    pynswork_ClassWorktype.o \
    pyns_nsworkmodule.o \
    pynsutil.o

all : ns.so

ns.so : $(PY_OBJS)
	$(CXX) $(LDFLAGS) -o $@ $^ $(LIBS)

namespace.o : namespace.hpp
