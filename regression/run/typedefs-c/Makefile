# Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and
# other Shroud Project Developers.
# See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
#
# #######################################################################
#
# run/typedefs-c
#
include $(top)/regression/run/defaults.mk

INCLUDE = \
    -I$(top)/regression/reference/typedefs-c \
    -I$(top)/regression/run/typedefs

VPATH = \
    $(top)/regression/reference/typedefs-c \
    $(top)/regression/run/typedefs \
    $(top)/regression/run/fruit

OBJS = \
    main.o \
    fruit.o \
    wrapftypedefs.o \
    typedefs.o \
    utiltypedefs.o

typedefs-c : $(OBJS)
	$(FC) $(FFLAGS) $^ -o $@ $(CLIBS)

main.o : main.f fruit.o wrapftypedefs.o
typedefs.o : typedefs.c typedefs.h

clean :
	rm -f $(OBJS) *.mod typedefs

main.o : fruit.o wrapftypedefs.o
