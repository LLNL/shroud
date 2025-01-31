# Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
# other Shroud Project Developers.
# See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
#
# #######################################################################
#
# run/pointers-cxx-c
# C wrappers only for a C++ library
#
include $(top)/regression/run/defaults.mk

testdir = pointers-cxx-c

INCLUDE = \
    -I$(top)/regression/reference/$(testdir) \
    -I$(top)/regression/run/pointers

VPATH = \
    $(top)/regression/reference/$(testdir) \
    ../cxx \
    $(top)/regression/run/$(testdir) \
    $(top)/regression/run/pointers \

C_OBJS = \
    pointers.o \
    wrappointers.o \
    utilpointers.o

pointers.o : pointers.cpp pointers.h
	$(CXX) $(CXXFLAGS) $(INCLUDE) -c -o $*.o $<

testc.o : testc.c wrappointers.h
testc : testc.o $(C_OBJS)
	$(CC) $(CFLAGS) $^ -o $@ $(CLIBS) -lm
clean :
	rm -f $(C_OBJS) testc
