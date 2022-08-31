# Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and
# other Shroud Project Developers.
# See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
# #######################################################################
#
# run/enum
#
ifeq ($(top),)
top := ../../..
endif 

include $(top)/regression/run/defaults.mk

INCLUDE = \
    -I$(top)/regression/reference/enum-c \
    -I$(top)/regression/run/enum-c \
    -I$(top)/regression/run/fruit

VPATH = \
    $(top)/regression/reference/enum-c \
    $(top)/regression/run/enum-c \
    $(top)/regression/run/fruit

C_OBJS = \
    enum.o
F_OBJS = \
    wrapfenum.o \
    fruit.o \
    main.o

enum-c : $(C_OBJS) $(F_OBJS)
	$(FC) $(FFLAGS) $^ -o $@ $(FLIBS)

clean :
	rm -f $(OBJS) *.mod enum-c

main.o : fruit.o wrapfenum.o

# useful to isolate load error with just C++ code
maincpp : maincpp.o 
	(CXX) $(CXXFLAGS) $(LIBS) $^ -o $@ $(CXXLIBS)

testc.o : testc.c enum.h
testc : testc.o $(C_OBJS)
	$(CC) $(CFLAGS) $^ -o $@ $(CLIBS)
