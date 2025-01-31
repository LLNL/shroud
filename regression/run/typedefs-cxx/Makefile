# Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
# other Shroud Project Developers.
# See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
#
# #######################################################################
#
# run/typedefs-cxx
#
include $(top)/regression/run/defaults.mk

INCLUDE = \
    -I$(top)/regression/reference/typedefs-cxx \
    -I$(top)/regression/run/typedefs

VPATH = \
    $(top)/regression/reference/typedefs-cxx \
    ../cxx \
    $(top)/regression/run/typedefs-cxx \
    $(top)/regression/run/typedefs \
    $(top)/regression/run/fruit

C_OBJS = \
    typedefs.o \
    wraptypedefs.o \
    utiltypedefs.o
F_OBJS = \
    main.o \
    fruit.o \
    wrapftypedefs.o

typedefs-cxx : $(F_OBJS) $(C_OBJS)
	$(FC) $(FFLAGS) $^ -o $@ $(CLIBS)

main.o : main.f fruit.o wrapftypedefs.o
typedefs.o : typedefs.cpp typedefs.h
	$(CXX) $(CXXFLAGS) $(INCLUDE) -c -o $*.o $<

testc.o : testc.c wraptypedefs.h
testc : testc.o $(C_OBJS)
	$(CC) $(CFLAGS) $^ -o $@ $(CLIBS) -lm
clean :
	rm -f $(OBJS) *.mod typedefs main.f

main.o : wrapftypedefs.o
