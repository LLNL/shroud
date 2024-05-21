# Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and
# other Shroud Project Developers.
# See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
# #######################################################################
#
# run/enum-c
#
ifeq ($(top),)
top := ../../..
endif 

refname = enum-c
runname = enum

include $(top)/regression/run/defaults.mk

INCLUDE = \
    -I$(top)/regression/reference/$(refname) \
    -I$(top)/regression/run/$(runname) \
    -I$(top)/regression/run/fruit

VPATH = \
    $(top)/regression/reference/$(refname) \
    $(top)/regression/run/$(runname) \
    $(top)/regression/run/fruit

C_OBJS = \
    enum.o \
    wrapenum.o
F_OBJS = \
    wrapfenum.o \
    fruit.o \
    main.o

$(refname) : $(C_OBJS) $(F_OBJS)
	$(FC) $(FFLAGS) $^ -o $@ $(FLIBS)

clean :
	rm -f $(OBJS) *.mod enum-c

main.o : fruit.o wrapfenum.o
wrapenum.o : wrapenum.h

# useful to isolate load error with just C++ code
maincpp : maincpp.o 
	(CXX) $(CXXFLAGS) $(LIBS) $^ -o $@ $(CXXLIBS)

testc.o : testc.c enum.h
testc : testc.o $(C_OBJS)
	$(CC) $(CFLAGS) $^ -o $@ $(CLIBS)
