# Copyright Shroud Project Developers. See LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
# #######################################################################
#
# run/tr29113
#
ifeq ($(top),)
top := ../../..
endif 

testdir = tr29113

INCLUDE = \
    -I$(top)/regression/reference/tr29113 \
    -I$(top)/regression/run/tr29113

VPATH = \
    $(top)/regression/reference/tr29113 \
    $(top)/regression/run/tr29113 \
    $(top)/regression/run/fruit

C_OBJS = \
    tr29113.o \
    utiltr29113.o \
    wraptr29113.o
F_OBJS = \
    main.o \
    fruit.o \
    wrapftr29113.o

ifeq ($(compiler),gcc)
FFLAGS += -std=f2008ts
endif

include $(top)/regression/run/defaults.mk


tr29113.o : tr29113.hpp
wrapTr29113.o : wraptr29113.h tr29113.hpp
main.o : fruit.o wrapftr29113.o

# useful to isolate load error with just C++ code
maincpp : maincpp.o wrapTr29113.o tr29113.o
	$(CXX) $(CXXFLAGS) $(LIBS) $^ -o $@ $(CXXLIBS)
