# Copyright Shroud Project Developers. See LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
#
# #######################################################################
#
#  Compile the python cxxlibrary module
#
testdir = cxxlibrary
pymodule = cxxlibrary.so

AS_SHARED := TRUE
include $(top)/regression/run/defaults.mk

INCLUDE = \
    -I$(top)/regression/reference/cxxlibrary \
    -I$(top)/regression/run/cxxlibrary \
    $(PYTHON_INC)

VPATH = \
    $(top)/regression/reference/cxxlibrary \
    $(top)/regression/run/cxxlibrary

PY_OBJS = \
    cxxlibrary.o \
    pyCstruct1_clstype.o \
    pycxxlibrarymodule.o \
    pycxxlibrary_structnsmodule.o \
    pycxxlibraryutil.o

all : cxxlibrary.so

cxxlibrary.so : $(PY_OBJS)
	$(CXX) $(LDFLAGS) -o $@ $^ $(LIBS)

cxxlibrary.o : cxxlibrary.hpp
