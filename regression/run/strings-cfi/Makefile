# Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and
# other Shroud Project Developers.
# See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
#
# #######################################################################
#
# run/strings-cfi - Futher Interoperability with C.
# Use test from run/strings.
#
include $(top)/regression/run/defaults.mk

INCLUDE = \
    -I$(top)/regression/reference/strings-cfi \
    -I$(top)/regression/run \
    -I$(top)/regression/run/strings \
    -I$(top)/regression/run/fruit

VPATH = \
    $(top)/regression/reference/strings-cfi \
    $(top)/regression/run/strings \
    $(top)/regression/run/fruit

OBJS = \
    strings.o \
    wrapfstrings.o \
    wrapstrings.o \
    fruit.o \
    main.o

strings-cfi : $(OBJS)
	$(FC) $(FFLAGS) $^ -o $@ $(FLIBS)

clean :
	rm -f $(OBJS) *.mod strings

main.o : fruit.o wrapfstrings.o
