# Copyright (c) 2017-2020, Lawrence Livermore National Security, LLC and
# other Shroud Project Developers.
# See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
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
