# Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
# other Shroud Project Developers.
# See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
#
# #######################################################################
#
# run/arrayclass
#
testdir = arrayclass

INCLUDE = \
    -I$(top)/regression/reference/arrayclass \
    -I$(top)/regression/run/arrayclass

VPATH = \
    $(top)/regression/reference/arrayclass \
    $(top)/regression/run/arrayclass \
    $(top)/regression/run/fruit

#    arrayclass.o   # This file is not created
C_OBJS = \
    wrapArrayWrapper.o \
    utilarrayclass.o
F_OBJS = \
    main.o \
    fruit.o \
    wrapfarrayclass.o

include $(top)/regression/run/defaults.mk

main.o : fruit.o wrapfarrayclass.o
