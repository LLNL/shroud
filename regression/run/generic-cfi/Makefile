# Copyright Shroud Project Developers. See LICENSE file for details.
#
# "SPDX-License-Identifier: (BSD-3-Clause)"
#
# run/generic-cfi - Futher Interoperability with C.
# Use test from run/generic.
#
testdir = generic-cfi

INCLUDE = \
    -I$(top)/regression/reference/generic-cfi \
    -I$(top)/regression/run/generic

VPATH = \
    $(top)/regression/reference/generic-cfi \
    $(top)/regression/run/generic \
    $(top)/regression/run/fruit

C_OBJS = \
    generic.o \
    helper.o \
    wrapgeneric.o
F_OBJS = \
    main.o \
    fruit.o \
    wrapfgeneric.o

include $(top)/regression/run/defaults.mk

helper.o : generic.h typesgeneric.h
generic.o : generic.h
wrapgeneric.o : generic.h
main.o : fruit.o wrapfgeneric.o


