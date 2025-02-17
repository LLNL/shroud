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

testdir = ownership

INCLUDE = \
    -I$(top)/regression/reference/ownership \
    -I$(top)/regression/run/ownership

VPATH = \
    $(top)/regression/reference/ownership \
    $(top)/regression/run/ownership \
    $(top)/regression/run/fruit

C_OBJS = \
    ownership.o \
    utilownership.o \
    wrapownership.o \
    wrapClass1.o
F_OBJS = \
    main.o \
    fruit.o \
    wrapfownership.o

include $(top)/regression/run/defaults.mk

ownership.o : ownership.hpp
wrapOwnership.o : wrapownership.h ownership.hpp
main.o : fruit.o wrapfownership.o

# useful to isolate load error with just C++ code
maincpp : maincpp.o ownership.o
	$(CXX) $(CXXFLAGS) $^ -o $@ $(CXXLIBS)
