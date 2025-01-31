# Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
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
    -I$(top)/regression/run/funptr

VPATH = \
    $(top)/regression/reference/funptr-cxx \
    ../cxx \
    $(top)/regression/run/funptr-cxx \
    $(top)/regression/run/funptr \
    $(top)/regression/run/fruit

C_OBJS = \
    funptr.o \
    wrapfunptr.o
F_OBJS = \
    wrapffunptr.o \
    fruit.o \
    main.o

funptr-cxx : $(C_OBJS) $(F_OBJS)
	$(FC) $(FFLAGS) $^ -o $@ $(CLIBS)

testc.o : testc.c wrapfunptr.h
testc : testc.o $(C_OBJS)
	$(CC) $(CFLAGS) $^ -o $@ $(CLIBS) -lm

clean :
	rm -f $(C_OBJS) $(F_OBJS) *.mod funptr-cxx

main.o : fruit.o wrapffunptr.o

funptr.o : funptr.cpp funptr.h
	$(CXX) $(CXXFLAGS) $(INCLUDE) -c -o $*.o $<
wrapfunptr.o : wrapfunptr.h funptr.h
