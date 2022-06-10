# Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and
# other Shroud Project Developers.
# See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
# #######################################################################
#
# run/overload64
#
ifeq ($(top),)
top := ../../..
endif 

include $(top)/regression/run/defaults.mk

TEST_CFLAGS   = -DUSE_64BIT_INDEXTYPE
TEST_CXXFLAGS = -DUSE_64BIT_INDEXTYPE
TEST_FFLAGS   = -DUSE_64BIT_INDEXTYPE

INCLUDE = \
    -I$(top)/regression/reference/overload \
    -I$(top)/regression/run/overload \
    -I$(top)/regression/run/fruit

VPATH = \
    $(top)/regression/reference/overload \
    $(top)/regression/run/overload \
    $(top)/regression/run/fruit

C_OBJS = \
    overload.o \
    utiloverload.o \
    wrapoverload.o
F_OBJS = \
    wrapfoverload.o \
    fruit.o \
    main.o

overload64 : $(C_OBJS) $(F_OBJS)
	$(FC) $(FFLAGS) $^ -o $@ $(FLIBS)

clean :
	rm -f $(OBJS) *.mod overload

overload.o : overload.cpp overload.hpp
wrapoverload.o : wrapoverload.h overload.hpp
main.o : fruit.o wrapfoverload.o

# useful to isolate load error with just C++ code
maincpp : maincpp.o overload.o
	$(CXX) $(CXXFLAGS) $(LIBS) $^ -o $@ $(CXXLIBS)

testc.o : testc.c wrapoverload.h overload.hpp
testc : testc.o $(C_OBJS)
	$(CC) $(CFLAGS) $^ -o $@ $(CLIBS)
