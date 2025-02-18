# Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
# other Shroud Project Developers.
# See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
# #######################################################################
#
# run/enum-c
#
ifeq ($(top),)
top := ../../..
endif

testdir = enum-c
runname = enum

INCLUDE = \
    -I$(top)/regression/reference/$(testdir) \
    -I$(top)/regression/run/$(runname)

VPATH = \
    $(top)/regression/reference/$(testdir) \
    $(top)/regression/run/$(runname) \
    $(top)/regression/run/fruit

C_OBJS = \
    enum.o \
    wrapenum.o
F_OBJS = \
    wrapfenum.o \
    fruit.o \
    main.o

include $(top)/regression/run/defaults.mk

main.o : fruit.o wrapfenum.o
wrapenum.o : wrapenum.h

# useful to isolate load error with just C++ code
maincpp : maincpp.o 
	(CXX) $(CXXFLAGS) $(LIBS) $^ -o $@ $(CXXLIBS)

testc.o : testc.c enum.h
