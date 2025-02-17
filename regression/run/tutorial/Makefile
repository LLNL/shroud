# Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
# other Shroud Project Developers.
# See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
# #######################################################################
#
# run/tutorial
#
ifeq ($(top),)
top := ../../..
endif 

testdir = tutorial

INCLUDE = \
    -I$(top)/regression/reference/tutorial \
    -I$(top)/regression/run/tutorial

VPATH = \
    $(top)/regression/reference/tutorial \
    $(top)/regression/run/tutorial \
    $(top)/regression/run/fruit

C_OBJS = \
    tutorial.o \
    utilTutorial.o \
    wrapTutorial.o
F_OBJS = \
    fruit.o \
    main.o \
    wrapftutorial.o

include $(top)/regression/run/defaults.mk

tutorial.o : tutorial.cpp tutorial.hpp
wrapTutorial.o : wrapTutorial.cpp wrapTutorial.h tutorial.hpp
main.o : fruit.o wrapftutorial.o

# useful to isolate load error with just C++ code
maincpp : maincpp.o tutorial.o
	$(CXX) $(CXXFLAGS) $(LIBS) $^ -o $@ $(CXXLIBS)

testc.o : testc.c wrapTutorial.h tutorial.hpp
