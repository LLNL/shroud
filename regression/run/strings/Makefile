# Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and
# other Shroud Project Developers.
# See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
#
# #######################################################################
#
# run/strings
#
include $(top)/regression/run/defaults.mk

TEST_FFLAGS = -DTEST_C_WRAPPER

INCLUDE = \
    -I$(top)/regression/reference/strings \
    -I$(top)/regression/run \
    -I$(top)/regression/run/strings \
    -I$(top)/regression/run/fruit

VPATH = \
    $(top)/regression/reference/strings \
    $(top)/regression/run/strings \
    $(top)/regression/run/fruit

OBJS = \
    strings.o \
    utilstrings.o \
    wrapfstrings.o \
    wrapstrings.o \
    fruit.o \
    main.o

strings : $(OBJS)
	$(FC) $(FFLAGS) $^ -o $@ $(FLIBS)

clean :
	rm -f $(OBJS) *.mod strings

main.o : fruit.o wrapfstrings.o
