# Copyright Shroud Project Developers. See LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
#
# #######################################################################
#
# run/preprocess
#
testdir = preprocess

INCLUDE = \
    -I$(top)/regression/reference/preprocess \
    -I$(top)/regression/run/preprocess

VPATH = \
    $(top)/regression/reference/preprocess \
    $(top)/regression/run/preprocess \
    $(top)/regression/run/fruit

C_OBJS = \
    wrapUser1.o \
    wrapUser2.o \
    utilpreprocess.o
F_OBJS = \
    main.o \
    fruit.o \
    wrapfpreprocess.o

include $(top)/regression/run/defaults.mk

wrappreprocess.o : preprocess.hpp
wrapUser1.o : wrapUser1.h preprocess.hpp wrapUser1.h
main.o : fruit.o wrapfpreprocess.o wrappreprocess.o
