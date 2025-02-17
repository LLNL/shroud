# Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
# other Shroud Project Developers.
# See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
#
# #######################################################################
#
# run/pointers-c
#
testdir = pointers-c
#TEST_FFLAGS = -DTEST_C_WRAPPER

INCLUDE = \
    -I$(top)/regression/reference/$(testdir) \
    -I$(top)/regression/run/pointers

VPATH = \
    $(top)/regression/reference/$(testdir) \
    $(top)/regression/run/pointers \
    $(top)/regression/run/fruit

C_OBJS = \
    pointers.o \
    wrappointers.o \
    utilpointers.o
F_OBJS = \
    main.o \
    fruit.o \
    wrapfpointers.o

include $(top)/regression/run/defaults.mk

main.o : fruit.o wrapfpointers.o
