# Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and
# other Shroud Project Developers.
# See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
# #######################################################################
#
# run/defaultarg64
#
ifeq ($(top),)
top := ../../..
endif 

include $(top)/regression/run/defaults.mk

TEST_CFLAGS   = -DINDEXTYPE_SIZE=64
TEST_CXXFLAGS = -DINDEXTYPE_SIZE=64
TEST_FFLAGS   = -DINDEXTYPE_SIZE=64

INCLUDE = \
    -I$(top)/regression/reference/defaultarg \
    -I$(top)/regression/run/defaultarg \
    -I$(top)/regression/run/fruit

VPATH = \
    $(top)/regression/reference/defaultarg \
    $(top)/regression/run/defaultarg \
    $(top)/regression/run/fruit

C_OBJS = \
    defaultarg.o \
    utildefaultarg.o \
    wrapdefaultarg.o \
    wrapClass1.o
F_OBJS = \
    wrapfdefaultarg.o \
    fruit.o \
    main.o

defaultarg64 : $(C_OBJS) $(F_OBJS)
	$(FC) $(FFLAGS) $^ -o $@ $(FLIBS)

clean :
	rm -f $(OBJS) *.mod defaultarg

defaultarg.o : defaultarg.cpp defaultarg.hpp
wrapdefaultarg.o : wrapdefaultarg.h defaultarg.hpp
wrapClass1.o : wrapClass1.cpp defaultarg.hpp wrapClass1.h
main.o : fruit.o wrapfdefaultarg.o

# useful to isolate load error with just C++ code
maincpp : maincpp.o defaultarg.o
	$(CXX) $(CXXFLAGS) $(LIBS) $^ -o $@ $(CXXLIBS)

testc.o : testc.c wrapdefaultarg.h defaultarg.hpp
testc : testc.o $(C_OBJS)
	$(CC) $(CFLAGS) $^ -o $@ $(CLIBS)
