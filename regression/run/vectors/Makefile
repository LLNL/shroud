# Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and
# other Shroud Project Developers.
# See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
#
# #######################################################################
#
# run/vectors
#
include $(top)/regression/run/defaults.mk

INCLUDE = \
    -I$(top)/regression/reference/vectors \
    -I$(top)/regression/run/vectors

VPATH = \
    $(top)/regression/reference/vectors \
    $(top)/regression/run/vectors \
    $(top)/regression/run/fruit

C_OBJS = \
    vectors.o \
    wrapvectors.o \
    utilvectors.o
F_OBJS = \
    wrapfvectors.o \
    fruit.o \
    main.o

vectors : $(C_OBJS) $(F_OBJS)
	$(FC) $(FFLAGS)  $^ -o $@ $(FLIBS)

clean :
	rm -f $(OBJS) *.mod vectors

vectors.o : vectors.hpp
wrapvectors.o : wrapvectors.h vectors.hpp
main.o : fruit.o wrapfvectors.o

# useful to isolate load error with just C++ code
maincpp : maincpp.o wrapvectors.o vectors.o
	$(CXX) $(CXXFLAGS) $^ -o $@ $(CLIBS)

testc.o : testc.c wrapvectors.h vectors.hpp
testc : testc.o $(C_OBJS)
	$(CC) $(CFLAGS) $^ -o $@ $(CLIBS)
