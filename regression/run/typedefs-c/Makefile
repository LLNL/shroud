# Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
# other Shroud Project Developers.
# See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
#
# #######################################################################
#
# run/typedefs-c
#
testdir = typedefs-c

INCLUDE = \
    -I$(top)/regression/reference/typedefs-c \
    -I$(top)/regression/run/typedefs

VPATH = \
    $(top)/regression/reference/typedefs-c \
    $(top)/regression/run/typedefs \
    $(top)/regression/run/fruit

C_OBJS = \
    typedefs.o \
    wraptypedefs.o \
    utiltypedefs.o
F_OBJS = \
    main.o \
    fruit.o \
    wrapftypedefs.o

include $(top)/regression/run/defaults.mk

main.o : main.f fruit.o wrapftypedefs.o
typedefs.o : typedefs.c typedefs.h

main.o : fruit.o wrapftypedefs.o
