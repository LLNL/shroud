# Copyright (c) 2017-2021, Lawrence Livermore National Security, LLC and
# other Shroud Project Developers.
# See the top-level COPYRIGHT file for details.
#
# "SPDX-License-Identifier: (BSD-3-Clause)"
#
# run/generic-cfi - Futher Interoperability with C.
# Use test from run/generic.
#
include $(top)/regression/run/defaults.mk

INCLUDE = \
    -I$(top)/regression/reference/generic-cfi \
    -I$(top)/regression/run/generic \
    -I$(top)/regression/run/fruit

VPATH = \
    $(top)/regression/reference/generic-cfi \
    $(top)/regression/run/generic \
    $(top)/regression/run/fruit

OBJS = \
    generic.o \
    helper.o \
    wrapfgeneric.o \
    wrapgeneric.o \
    fruit.o \
    main.o

generic-cfi : $(OBJS)
	$(FC) $(FFLAGS) $^ -o $@ $(FLIBS)

clean :
	rm -f $(OBJS) *.mod generic

helper.o : generic.h typesgeneric.h
generic.o : generic.h
wrapgeneric.o : generic.h
main.o : fruit.o wrapfgeneric.o


