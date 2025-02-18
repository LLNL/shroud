# Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
# other Shroud Project Developers.
# See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
# #######################################################################
#
# run/forward
#
testdir = forward

INCLUDE = \
    -I$(top)/regression/reference/forward \
    -I$(top)/regression/run/forward

#    -I$(top)/regression/run/tutorial


VPATH = \
    $(top)/regression/reference/forward \
    $(top)/regression/run/forward \
    $(top)/regression/run/fruit

C_OBJS = \
    forward.o \
    wrapforward.o \
    wrapClass2.o
F_OBJS = \
    main.o \
    fruit.o \
    wrapfforward.o \
    struct_mod.o \
    tutorial_mod.o

include $(top)/regression/run/defaults.mk

# 'other' packages
struct_mod.o : struct_mod.f
tutorial_mod.o : tutorial_mod.f

wrapforward.o : wrapforward.h
wrapfforward.o : struct_mod.o tutorial_mod.o
forward.o : forward.hpp
wrapClass1.o : wrapClass2.h forward.hpp
main.o : fruit.o wrapfforward.o struct_mod.o

# useful to isolate load error with just C++ code
maincpp : maincpp.o wrapClass2.o forward.o
	$(CXX) $(CXXFLAGS) $^ -o $@ $(CXXLIBS)
