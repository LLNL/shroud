# Copyright (c) 2017-2021, Lawrence Livermore National Security, LLC and
# other Shroud Project Developers.
# See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
# #######################################################################
#
# run/namespace
#
ifeq ($(top),)
top := ../../..
endif 

include $(top)/regression/run/defaults.mk

INCLUDE = \
    -I$(top)/regression/reference/namespace \
    -I$(top)/regression/run/namespace \
    -I$(top)/regression/run/fruit

VPATH = \
    $(top)/regression/reference/namespace \
    $(top)/regression/run/namespace \
    $(top)/regression/run/fruit

C_OBJS = \
    namespace.o \
    utilns.o \
    wrapfns_outer.o \
    wrapns.o \
    wrapns_outer.o
F_OBJS= \
    wrapfns.f \
    fruit.o \
    main.o

namespace : $(C_OBJS) $(F_OBJS)
	$(FC) $(FFLAGS) $^ -o $@ $(FLIBS)

clean :
	rm -f $(OBJS) *.mod namespace

namespace.o : namespace.hpp
wrapns.o : namespace.hpp
wrapns_outer.o : namespace.hpp
main.o : wrapfns.o wrapfns_outer.o

testc.o : testc.c wrapns.h wrapns_outer.h
testc : testc.o $(C_OBJS)
	$(CC) $(CFLAGS) $^ -o $@ $(CLIBS)
