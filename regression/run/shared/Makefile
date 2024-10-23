# Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and
# other Shroud Project Developers.
# See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
# #######################################################################
#
# run/shared
#
ifeq ($(top),)
top := ../../..
endif 

include $(top)/regression/run/defaults.mk

INCLUDE = \
    -I$(top)/regression/reference/shared \
    -I$(top)/regression/run/shared \
    -I$(top)/regression/run/fruit

VPATH = \
    $(top)/regression/reference/shared \
    $(top)/regression/run/shared \
    $(top)/regression/run/fruit

C_OBJS = \
    utilShared.o \
    wrapObject.o
F_OBJS = \
    wrapfshared.o \
    fruit.o \
    main.o

shared : $(C_OBJS) $(F_OBJS)
	$(FC) $(FFLAGS) $^ -o $@ $(FLIBS)

clean :
	rm -f $(OBJS) *.mod shared

#shared.o : shared.cpp shared.hpp
utilShared.o : utilShared.cpp shared.hpp typesShared.h
wrapObject.o : wrapObject.cpp shared.hpp wrapObject.h
main.o : fruit.o wrapfshared.o

# useful to isolate load error with just C++ code
maincpp : maincpp.o
	$(CXX) $(CXXFLAGS) $(LIBS) $^ -o $@ $(CXXLIBS)

testc.o : testc.c wrapShared.h shared.hpp
testc : testc.o $(C_OBJS)
	$(CC) $(CFLAGS) $^ -o $@ $(CLIBS)
