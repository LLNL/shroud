# Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and
# other Shroud Project Developers.
# See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
#
# #######################################################################
#
# run/pointers-cfi
#
include $(top)/regression/run/defaults.mk

testdir = pointers-cfi

INCLUDE = \
    -I$(top)/regression/reference/$(testdir) \
    -I$(top)/regression/run/pointers

VPATH = \
    $(top)/regression/reference/$(testdir) \
    ../cxx \
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

main.o : main.f fruit.o wrapfpointers.o
pointers.o : pointers.cpp pointers.h
	$(CXX) $(CXXFLAGS) $(INCLUDE) -c -o $*.o $<

clean :
	rm -f $(OBJS) *.mod $(testdir)

main.o : wrapfpointers.o
