# Copyright Shroud Project Developers. See LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
#
# #######################################################################
#
# run/pointers-cxx-c
# C wrappers only for a C++ library
#
testdir = pointers-cxx-c

INCLUDE = \
    -I$(top)/regression/reference/$(testdir) \
    -I$(top)/regression/run/pointers

VPATH = \
    $(top)/regression/reference/$(testdir) \
    ../cxx \
    $(top)/regression/run/$(testdir) \
    $(top)/regression/run/pointers \

C_OBJS = \
    pointers.o \
    wrappointers.o \
    utilpointers.o

include $(top)/regression/run/defaults.mk

pointers.o : pointers.cpp pointers.h
	$(CXX) $(CXXFLAGS) $(INCLUDE) -c -o $*.o $<

testc.o : testc.c wrappointers.h
