# Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and
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
    ../cxx \
    $(top)/regression/run/pointers-cxx \
    $(top)/regression/run/pointers \
    $(top)/regression/run/fruit

C_OBJS = \
    pointers.o \
    wrappointers.o \
    utilpointers.o
F_OBJS = \
    main.o \
    fruit.o \
    wrapfpointers.o

pointers-cxx : $(F_OBJS) $(C_OBJS)
	$(FC) $(FFLAGS) $^ -o $@ $(CLIBS)

main.o : main.f fruit.o wrapfpointers.o
pointers.o : pointers.cpp pointers.h
	$(CXX) $(CXXFLAGS) $(INCLUDE) -c -o $*.o $<

testc.o : testc.c wrappointers.h
testc : testc.o $(C_OBJS)
	$(CC) $(CFLAGS) $^ -o $@ $(CLIBS) -lm
clean :
	rm -f $(OBJS) *.mod pointers main.f pointers.cpp

main.o : wrapfpointers.o
