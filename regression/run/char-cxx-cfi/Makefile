# Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
# other Shroud Project Developers.
# See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
#
# #######################################################################
#
# run/char-cxx-cfi
#
include $(top)/regression/run/defaults.mk

testdir = char-cxx-cfi
#TEST_FFLAGS = -DTEST_C_WRAPPER

INCLUDE = \
    -I$(top)/regression/reference/$(testdir) \
    -I$(top)/regression/run \
    -I$(top)/regression/run/char

VPATH = \
    $(top)/regression/reference/$(testdir) \
    ../cxx \
    $(top)/regression/run/char \
    $(top)/regression/run/fruit

C_OBJS = \
    char.o \
    wrapchar.o \
    utilchar.o
F_OBJS = \
    main.o \
    fruit.o \
    wrapfchar.o

$(testdir) : $(F_OBJS) $(C_OBJS)
	$(FC) $(FFLAGS) $^ -o $@ $(CLIBS)

main.o : main.f fruit.o wrapfchar.o
char.o : char.cpp char.h
	$(CXX) $(CXXFLAGS) $(INCLUDE) -c -o $*.o $<

testc.o : testc.c wrapchar.h
testc : testc.o $(C_OBJS)
	$(CC) $(CFLAGS) $^ -o $@ $(CLIBS) -lm
clean :
	rm -f $(C_OBJS) $(F_OBJS) *.mod $(testdir) testc

main.o : wrapfchar.o
