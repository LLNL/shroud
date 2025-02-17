# Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
# other Shroud Project Developers.
# See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
#
# #######################################################################
#
# run/strings
#
testdir = strings

#TEST_FFLAGS = -DTEST_C_WRAPPER

INCLUDE = \
    -I$(top)/regression/reference/strings \
    -I$(top)/regression/run \
    -I$(top)/regression/run/strings

VPATH = \
    $(top)/regression/reference/strings \
    $(top)/regression/run/strings \
    $(top)/regression/run/fruit

C_OBJS = \
    strings.o \
    utilstrings.o \
    wrapstrings.o
F_OBJS = \
    main.o \
    fruit.o \
    wrapfstrings.o

include $(top)/regression/run/defaults.mk

main.o : fruit.o wrapfstrings.o

testc.o : testc.c wrapstrings.h
