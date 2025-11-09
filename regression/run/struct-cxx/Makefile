# Copyright Shroud Project Developers. See LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
#
# #######################################################################
#
# run/struct-cxx
#
ifndef top
top = ../../..
endif

testdir = struct-cxx

INCLUDE = \
    -I$(top)/regression/reference/struct-cxx \
    -I$(top)/regression/run/struct

VPATH = \
    $(top)/regression/reference/struct-cxx \
    ../cxx \
    $(top)/regression/run/struct-cxx \
    $(top)/regression/run/struct \
    $(top)/regression/run/fruit

C_OBJS = \
    struct.o \
    wrapstruct.o \
    wrapCstruct_as_class.o \
    wrapCstruct_as_subclass.o \
    wrapCstruct_as_class2.o \
    utilstruct.o
F_OBJS = \
    main.o \
    fruit.o \
    wrapfstruct.o

include $(top)/regression/run/defaults.mk

struct.o : struct.cpp struct.h
	$(CXX) $(CXXFLAGS) $(INCLUDE) -c -o $*.o $<
wrapstruct.o : wrapstruct.h struct.h
main.o : fruit.o wrapfstruct.o

testc.o : testc.c wrapstruct.h struct.h
