# Copyright (c) 2017-2021, Lawrence Livermore National Security, LLC and
# other Shroud Project Developers.
# See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
#
# run/clibrary
#
include $(top)/regression/run/defaults.mk

INCLUDE = \
    -I$(top)/regression/reference/clibrary \
    -I$(top)/regression/run/clibrary \
    -I$(top)/regression/run/fruit

VPATH = \
    $(top)/regression/reference/clibrary \
    $(top)/regression/run/clibrary \
    $(top)/regression/run/fruit

OBJS = \
    clibrary.o \
    wrapfclibrary.o \
    wrapClibrary.o \
    fruit.o \
    main.o

clibrary : $(OBJS)
	$(FC) $(FFLAGS) $^ -o $@ $(CLIBS)

clean :
	rm -f $(OBJS) *.mod clibrary

main.o : wrapfclibrary.o
