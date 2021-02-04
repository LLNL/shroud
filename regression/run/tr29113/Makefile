# Copyright (c) 2017-2021, Lawrence Livermore National Security, LLC and
# other Shroud Project Developers.
# See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
# #######################################################################
#
# run/tr29113
#
ifeq ($(top),)
top := ../../..
endif 

include $(top)/regression/run/defaults.mk

INCLUDE = \
    -I$(top)/regression/reference/tr29113 \
    -I$(top)/regression/run/tr29113 \
    -I$(top)/regression/run/fruit

VPATH = \
    $(top)/regression/reference/tr29113 \
    $(top)/regression/run/tr29113 \
    $(top)/regression/run/fruit

C_OBJS = \
    tr29113.o \
    utiltr29113.o \
    wraptr29113.o
F_OBJS = \
    wrapftr29113.o \
    fruit.o \
    main.o

ifeq ($(compiler),gcc)
FFLAGS += -std=f2008ts
endif

tr29113 : $(C_OBJS) $(F_OBJS)
	$(FC) $(FFLAGS) $^ -o $@ $(FLIBS)

clean :
	rm -f $(OBJS) *.mod tr29113

tr29113.o : tr29113.hpp
wrapTr29113.o : wraptr29113.h tr29113.hpp
main.o : wrapftr29113.o

# useful to isolate load error with just C++ code
maincpp : maincpp.o wrapTr29113.o tr29113.o
	$(CXX) $(CXXFLAGS) $(LIBS) $^ -o $@ $(CXXLIBS)
