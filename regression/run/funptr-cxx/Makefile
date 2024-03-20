# Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and
# other Shroud Project Developers.
# See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
#
# run/funptr-cxx
#
include $(top)/regression/run/defaults.mk

INCLUDE = \
    -I$(top)/regression/reference/funptr-cxx \
    -I$(top)/regression/run/funptr \
    -I$(top)/regression/run/fruit

VPATH = \
    $(top)/regression/reference/funptr-cxx \
    ../cxx \
    $(top)/regression/run/funptr \
    $(top)/regression/run/fruit

OBJS = \
    funptr.o \
    wrapffunptr.o \
    wrapfunptr.o \
    fruit.o \
    main.o

funptr-cxx : $(OBJS)
	$(FC) $(FFLAGS) $^ -o $@ $(CLIBS)

clean :
	rm -f $(OBJS) *.mod funptr-cxx

main.o : fruit.o wrapffunptr.o
