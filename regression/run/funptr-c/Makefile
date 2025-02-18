# Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
# other Shroud Project Developers.
# See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
#
# run/funptr-c
#
testdir = funptr-c

INCLUDE = \
    -I$(top)/regression/reference/funptr-c \
    -I$(top)/regression/run/funptr

VPATH = \
    $(top)/regression/reference/funptr-c \
    $(top)/regression/run/funptr \
    $(top)/regression/run/fruit

C_OBJS = \
    funptr.o
F_OBJS = \
    main.o \
    fruit.o \
    wrapffunptr.o

include $(top)/regression/run/defaults.mk

main.o : fruit.o wrapffunptr.o
funptr.o : funptr.c funptr.h
wrapfunptr.o : wrapfunptr.h funptr.h
