# Copyright Shroud Project Developers. See LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
#
# #######################################################################
#
#  Compile the python vectors module
#
testdir = vectors-numpy
pymodule = vectors.so

AS_SHARED := TRUE
include $(top)/regression/run/defaults.mk

INCLUDE = \
    -I$(top)/regression/reference/vectors-numpy \
    -I$(top)/regression/run/vectors \
    $(PYTHON_INC)

VPATH = \
    $(top)/regression/reference/vectors-numpy \
    $(top)/regression/run/vectors \
    $(top)/regression/run/vectors-numpy/python

PY_OBJS = \
    vectors.o \
    pyvectorsmodule.o \
    pyvectorsutil.o

all : vectors.so #simple

vectors.so : $(PY_OBJS)
	$(CXX) $(LDFLAGS) -o $@ $^ $(LIBS)

simple : testpython.o $(PY_OBJS)
	$(CXX) -pthread -o $@ $^ $(PYTHON_LIB)
#	g++ -pthread -o $@ $^ $(PYTHON_LIB)

vectors.o : vectors.hpp
pyvectorsmodule.o : pyvectorsmodule.hpp
pyvectorsutil.o : pyvectorsmodule.hpp
