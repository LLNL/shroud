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
# run/ownership
#
ifndef top
top = ../..
endif

include $(top)/regression/run/defaults.mk

INCLUDE = \
    -I$(top)/regression/reference/ownership \
    -I$(top)/regression/run/ownership \
    -I$(top)/regression/run/fruit

VPATH = \
    $(top)/regression/reference/ownership \
    $(top)/regression/run/ownership \
    $(top)/regression/run/fruit

OBJS = \
    ownership.o \
    wrapfownership.o \
    wrapownership.o \
    wrapClass1.o \
    fruit.o \
    main.o

ownership : $(OBJS)
	$(FC) $(FFLAGS) $^ -o $@ $(FLIBS)

clean :
	rm -f $(OBJS) *.mod ownership maincpp maincpp.o

ownership.o : ownership.hpp
wrapOwnership.o : wrapownership.h ownership.hpp
main.o : wrapfownership.o

# useful to isolate load error with just C++ code
maincpp : maincpp.o ownership.o
	$(CXX) $(CXXFLAGS) $^ -o $@ $(CXXLIBS)
