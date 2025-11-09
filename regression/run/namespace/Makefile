# Copyright Shroud Project Developers. See LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
# #######################################################################
#
# run/namespace
#
ifeq ($(top),)
top := ../../..
endif 

testdir = namespace

INCLUDE = \
    -I$(top)/regression/reference/namespace \
    -I$(top)/regression/run/namespace

VPATH = \
    $(top)/regression/reference/namespace \
    $(top)/regression/run/namespace \
    $(top)/regression/run/fruit

C_OBJS = \
    namespace.o \
    utilns.o \
    wrapns.o \
    wrapns_outer.o
F_OBJS= \
    main.o \
    fruit.o \
    wrapfns.o \
    wrapfns_outer.o

include $(top)/regression/run/defaults.mk

namespace.o : namespace.hpp
wrapns.o : namespace.hpp
wrapns_outer.o : namespace.hpp
main.o : fruit.o wrapfns.o wrapfns_outer.o

testc.o : testc.c wrapns.h wrapns_outer.h

