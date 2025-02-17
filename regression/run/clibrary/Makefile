# Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
# other Shroud Project Developers.
# See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
#
# run/clibrary
#
testdir = clibrary

INCLUDE = \
    -I$(top)/regression/reference/clibrary \
    -I$(top)/regression/run/clibrary

VPATH = \
    $(top)/regression/reference/clibrary \
    $(top)/regression/run/clibrary \
    $(top)/regression/run/fruit

C_OBJS = \
    clibrary.o \
    wrapClibrary.o
F_OBJS = \
    main.o \
    fruit.o \
    wrapfclibrary.o

include $(top)/regression/run/defaults.mk

main.o : fruit.o wrapfclibrary.o
