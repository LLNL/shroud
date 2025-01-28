# Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and
# other Shroud Project Developers.
# See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
#
# run/funptr-c
#
include $(top)/regression/run/defaults.mk

INCLUDE = \
    -I$(top)/regression/reference/funptr-c \
    -I$(top)/regression/run/funptr

VPATH = \
    $(top)/regression/reference/funptr-c \
    $(top)/regression/run/funptr \
    $(top)/regression/run/fruit

OBJS = \
    funptr.o \
    wrapffunptr.o \
    fruit.o \
    main.o

funptr-c : $(OBJS)
	$(FC) $(FFLAGS) $^ -o $@ $(CLIBS)

clean :
	rm -f $(OBJS) *.mod funptr-c

main.o : fruit.o wrapffunptr.o
funptr.o : funptr.c funptr.h
wrapfunptr.o : wrapfunptr.h funptr.h
