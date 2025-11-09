# Copyright Shroud Project Developers. See LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
#
# #######################################################################
#
testdir = pointers-cxx-force
#TEST_FFLAGS = -DTEST_C_WRAPPER

INCLUDE = \
    -I$(top)/regression/reference/$(testdir) \
    -I$(top)/regression/run/pointers

VPATH = \
    $(top)/regression/reference/$(testdir) \
    ../cxx \
    $(top)/regression/run/pointers \
    $(top)/regression/run/fruit

C_OBJS = \
    pointers.o \
    wrappointers.o \
    utilpointers.o
F_OBJS = \
    main.o \
    fruit.o \
    wrapfpointers.o

include $(top)/regression/run/defaults.mk

main.o : main.f fruit.o wrapfpointers.o
pointers.o : pointers.cpp pointers.h
	$(CXX) $(CXXFLAGS) $(INCLUDE) -c -o $*.o $<

testc.o : testc.c wrappointers.h
