# Copyright Shroud Project Developers. See LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
#
# #######################################################################
#
# run/strings-cfi - Futher Interoperability with C.
# Use test from run/strings.
#
testdir = strings-cfi

INCLUDE = \
    -I$(top)/regression/reference/strings-cfi \
    -I$(top)/regression/run \
    -I$(top)/regression/run/strings

VPATH = \
    $(top)/regression/reference/strings-cfi \
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
