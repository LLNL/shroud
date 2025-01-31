# Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
# other Shroud Project Developers.
# See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
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
    -I$(top)/regression/run/ownership

VPATH = \
    $(top)/regression/reference/ownership \
    $(top)/regression/run/ownership \
    $(top)/regression/run/fruit

OBJS = \
    ownership.o \
    utilownership.o \
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
main.o : fruit.o wrapfownership.o

# useful to isolate load error with just C++ code
maincpp : maincpp.o ownership.o
	$(CXX) $(CXXFLAGS) $^ -o $@ $(CXXLIBS)
