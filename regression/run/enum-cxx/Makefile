# Copyright Shroud Project Developers. See LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
# #######################################################################
#
# run/enum-cxx
#
ifeq ($(top),)
top := ../../..
endif 

testdir = enum-cxx
runname = enum

INCLUDE = \
    -I$(top)/regression/reference/$(testdir) \
    -I$(top)/regression/run/$(runname)

VPATH = \
    $(top)/regression/reference/$(testdir) \
    ../cxx \
    $(top)/regression/run/$(runname) \
    $(top)/regression/run/fruit

C_OBJS = \
    enum.o \
    wrapenum.o \
    utilenum.o
F_OBJS = \
    main.o \
    fruit.o \
    wrapfenum.o

include $(top)/regression/run/defaults.mk

main.o : fruit.o wrapfenum.o
enum.o : enum.cpp enum.h
	$(CXX) $(CXXFLAGS) $(INCLUDE) -c -o $*.o $<
wrapenum.o : wrapenum.h

# useful to isolate load error with just C++ code
maincpp : maincpp.o 
	(CXX) $(CXXFLAGS) $(LIBS) $^ -o $@ $(CXXLIBS)

testc.o : testc.c enum.h
