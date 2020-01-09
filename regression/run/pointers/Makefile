# Copyright (c) 2017-2020, Lawrence Livermore National Security, LLC and
# other Shroud Project Developers.
# See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
#
# #######################################################################
#
# run/pointers-numpy-cpp
#
include $(top)/regression/run/defaults.mk

INCLUDE = \
    -I$(top)/regression/reference/pointers-numpy-cpp \
    -I$(top)/regression/run/pointers \
    -I$(top)/regression/run/fruit

VPATH = \
    $(top)/regression/reference/pointers-numpy-cpp \
    $(top)/regression/run/pointers \
    $(top)/regression/run/fruit

OBJS = \
    pointers.o \
    wrapfpointers.o \
    wrappointers.o \
    fruit.o \
    main.o

pointers : $(OBJS)
	$(FC) $(FFLAGS) $^ -o $@ $(CLIBS)

clean :
	rm -f $(OBJS) *.mod pointers

main.o : wrapfpointers.o
