# Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
# other Shroud Project Developers.
# See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
#
# #######################################################################
#
# run/struct-c
#
testdir = struct-c

INCLUDE = \
    -I$(top)/regression/reference/struct-c \
    -I$(top)/regression/run/struct

VPATH = \
    $(top)/regression/reference/struct-c \
    $(top)/regression/run/struct \
    $(top)/regression/run/fruit

C_OBJS = \
    struct.o \
    wrapstruct.o \
    wrapCstruct_as_class.o \
    wrapCstruct_as_subclass.o
F_OBJS = \
    main.o \
    fruit.o \
    wrapfstruct.o

include $(top)/regression/run/defaults.mk

struct.o : struct.h
wrapstruct.o : wrapstruct.h struct.h
main.o : fruit.o wrapfstruct.o
