# Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and
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

testdir = pointers-c
#TEST_FFLAGS = -DTEST_C_WRAPPER

INCLUDE = \
    -I$(top)/regression/reference/$(testdir) \
    -I$(top)/regression/run/pointers

VPATH = \
    $(top)/regression/reference/$(testdir) \
    $(top)/regression/run/pointers \
    $(top)/regression/run/fruit

OBJS = \
    main.o \
    fruit.o \
    wrapfpointers.o \
    pointers.o \
    wrappointers.o \
    utilpointers.o

$(testdir) : $(OBJS)
	$(FC) $(FFLAGS) $^ -o $@ $(CLIBS)

clean :
	rm -f $(OBJS) *.mod $(testdir)

main.o : fruit.o wrapfpointers.o
