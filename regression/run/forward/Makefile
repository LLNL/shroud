# Copyright (c) 2017-2020, Lawrence Livermore National Security, LLC and
# other Shroud Project Developers.
# See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
# #######################################################################
#
# run/forward
#
include $(top)/regression/run/defaults.mk

INCLUDE = \
    -I$(top)/regression/reference/forward \
    -I$(top)/regression/run/forward \
    -I$(top)/regression/run/fruit

#    -I$(top)/regression/run/tutorial


VPATH = \
    $(top)/regression/reference/forward \
    $(top)/regression/run/forward \
    $(top)/regression/run/fruit

OBJS = \
    tutorial_mod.o \
    forward.o \
    wrapfforward.o \
    wrapClass2.o \
    fruit.o \
    main.o

forward : $(OBJS)
	$(FC) $(FFLAGS) $^ -o $@ $(FLIBS)

clean :
	rm -f $(OBJS) *.mod forward

# 'other' package
tutorial_mod.o : tutorial_mod.f

wrapfforward.o : tutorial_mod.o
forward.o : forward.hpp
wrapClass1.o : wrapClass2.h forward.hpp
main.o : wrapfforward.o

# useful to isolate load error with just C++ code
maincpp : maincpp.o wrapClass2.o forward.o
	$(CXX) $(CXXFLAGS) $^ -o $@ $(CXXLIBS)
