# Copyright Shroud Project Developers. See LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
#
# run/ccomplex
#
testdir = ccomplex

INCLUDE = \
    -I$(top)/regression/reference/ccomplex \
    -I$(top)/regression/run/ccomplex

VPATH = \
    $(top)/regression/reference/ccomplex \
    $(top)/regression/run/ccomplex \
    $(top)/regression/run/fruit

C_OBJS = \
    ccomplex.o \
    utilccomplex.o
F_OBJS = \
    main.o \
    fruit.o \
    wrapfccomplex.o

include $(top)/regression/run/defaults.mk

main.o : fruit.o wrapfccomplex.o
