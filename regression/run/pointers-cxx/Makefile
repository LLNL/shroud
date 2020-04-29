# Copyright (c) 2017-2020, Lawrence Livermore National Security, LLC and
# other Shroud Project Developers.
# See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
#
# #######################################################################
#
# run/pointers-cxx
#
include $(top)/regression/run/defaults.mk

INCLUDE = \
    -I$(top)/regression/reference/pointers-cxx \
    -I$(top)/regression/run/pointers

VPATH = \
    $(top)/regression/reference/pointers-cxx \
    $(top)/regression/run/fruit

OBJS = \
    pointers.o \
    wrapfpointers.o \
    wrappointers.o \
    fruit.o \
    main.o

pointers-cxx : $(OBJS)
	$(FC) $(FFLAGS) $^ -o $@ $(CLIBS)

# Copy all source files from run/pointers, need to avoid VPATH.
# Convert C file to C++
pointers.cpp : $(top)/regression/run/pointers/pointers.c
	cp $^ $@
main.f : $(top)/regression/run/pointers/main.f
	cp $< $@
main.o : main.f
pointers.o : pointers.cpp

clean :
	rm -f $(OBJS) *.mod pointers main.f pointers.cpp

main.o : wrapfpointers.o
