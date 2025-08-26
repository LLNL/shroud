# Copyright Shroud Project Developers. See LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
#
# #######################################################################
#
#  Compile the python preprocess module
#
testdir = preprocess
pymodule = preprocess.so

AS_SHARED := TRUE
include $(top)/regression/run/defaults.mk

INCLUDE = \
    -I$(top)/regression/reference/preprocess \
    -I$(top)/regression/run/preprocess \
    $(PYTHON_INC)

VPATH = \
    $(top)/regression/reference/preprocess \
    $(top)/regression/run/preprocess \
    $(top)/regression/run/preprocess/python

PY_OBJS = \
    pypreprocessmodule.o \
    pypreprocessutil.o \
    pyUser1type.o \
    pyUser2type.o

all : preprocess.so

preprocess.so : $(PY_OBJS)
	$(CXX) $(LDFLAGS) -o $@ $^ $(LIBS)

