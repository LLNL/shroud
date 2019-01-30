# Copyright (c) 2017-2019, Lawrence Livermore National Security, LLC. 
#
# Produced at the Lawrence Livermore National Laboratory 
#
# LLNL-CODE-738041.
#
# All rights reserved. 
#
# This file is part of Shroud.
#
# For details about use and distribution, please read LICENSE.
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
    wrapvectors.o \
    fruit.o \
    main.o

vectors : $(OBJS)
	$(FC) $(FFLAGS)  $^ -o $@ $(FLIBS)

clean :
	rm -f $(OBJS) *.mod vectors

vectors.o : vectors.hpp
wrapvectors.o : wrapvectors.h vectors.hpp
main.o : wrapfvectors.o

# useful to isolate load error with just C++ code
maincpp : maincpp.o wrapvectors.o vectors.o
	$(CXX) $(CXXFLAGS) $^ -o $@ $(CLIBS)
