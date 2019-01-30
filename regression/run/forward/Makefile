# Copyright (c) 2018-2019, Lawrence Livermore National Security, LLC. 
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
# run/forward
#
include $(top)/regression/run/defaults.mk

INCLUDE = \
    -I$(top)/regression/reference/forward \
    -I$(top)/regression/run/forward \
    -I$(top)/regression/run/tutorial \
    -I$(top)/regression/run/fruit

VPATH = \
    $(top)/regression/reference/forward \
    $(top)/regression/run/forward \
    $(top)/regression/run/fruit

OBJS = \
    forward.o \
    wrapfforward.o \
    wrapClass2.o \
    fruit.o \
    main.o

forward : $(OBJS)
	$(FC) $(FFLAGS) $^ -o $@ $(FLIBS)

clean :
	rm -f $(OBJS) *.mod forward

forward.o : forward.hpp
wrapClass1.o : wrapClass2.h forward.hpp
main.o : wrapfforward.o

# useful to isolate load error with just C++ code
maincpp : maincpp.o wrapClass2.o forward.o
	$(CXX) $(CXXFLAGS) $^ -o $@ $(CXXLIBS)
