# Copyright Shroud Project Developers. See LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
#
# #######################################################################
#
# run/char-cxx
#
testdir = char-cxx
#TEST_FFLAGS = -DTEST_C_WRAPPER

INCLUDE = \
    -I$(top)/regression/reference/$(testdir) \
    -I$(top)/regression/run \
    -I$(top)/regression/run/char

VPATH = \
    $(top)/regression/reference/$(testdir) \
    ../cxx \
    $(top)/regression/run/char \
    $(top)/regression/run/fruit

C_OBJS = \
    char.o \
    wrapchar.o \
    utilchar.o
F_OBJS = \
    main.o \
    fruit.o \
    wrapfchar.o

include $(top)/regression/run/defaults.mk

main.o : main.f fruit.o wrapfchar.o
char.o : char.cpp char.h
	$(CXX) $(CXXFLAGS) $(INCLUDE) -c -o $*.o $<

testc.o : testc.c wrapchar.h
