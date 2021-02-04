# Copyright (c) 2017-2021, Lawrence Livermore National Security, LLC and
# other Shroud Project Developers.
# See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
#
# #######################################################################
#
# run/pointers-c
#
include $(top)/regression/run/defaults.mk

INCLUDE = \
    -I$(top)/regression/reference/pointers-c \
    -I$(top)/regression/run/pointers

VPATH = \
    $(top)/regression/reference/pointers-c \
    $(top)/regression/run/pointers \
    $(top)/regression/run/fruit

OBJS = \
    pointers.o \
    wrapfpointers.o \
    wrappointers.o \
    fruit.o \
    main.o

pointers-c : $(OBJS)
	$(FC) $(FFLAGS) $^ -o $@ $(CLIBS)

clean :
	rm -f $(OBJS) *.mod pointers

main.o : wrapfpointers.o
