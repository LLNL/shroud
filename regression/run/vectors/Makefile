# Copyright (c) 2017-2021, Lawrence Livermore National Security, LLC and
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
    -I$(top)/regression/run/vectors \
    -I$(top)/regression/run/fruit

VPATH = \
    $(top)/regression/reference/vectors \
    $(top)/regression/run/vectors \
    $(top)/regression/run/fruit

OBJS = \
    vectors.o \
    wrapfvectors.o \
    utilvectors.o \
    wrapvectors.o \
    fruit.o \
    main.o

vectors : $(OBJS)
	$(FC) $(FFLAGS)  $^ -o $@ $(FLIBS)

clean :
	rm -f $(OBJS) *.mod vectors

vectors.o : vectors.hpp
wrapvectors.o : wrapvectors.h vectors.hpp
main.o : fruit.o wrapfvectors.o

# useful to isolate load error with just C++ code
maincpp : maincpp.o wrapvectors.o vectors.o
	$(CXX) $(CXXFLAGS) $^ -o $@ $(CLIBS)
