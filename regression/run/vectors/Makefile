# Copyright Shroud Project Developers. See LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
#
# #######################################################################
#
# run/vectors
#
testdir = vectors

INCLUDE = \
    -I$(top)/regression/reference/vectors \
    -I$(top)/regression/run/vectors

VPATH = \
    $(top)/regression/reference/vectors \
    $(top)/regression/run/vectors \
    $(top)/regression/run/fruit

C_OBJS = \
    vectors.o \
    wrapvectors.o \
    utilvectors.o
F_OBJS = \
    wrapfvectors.o \
    fruit.o \
    main.o

include $(top)/regression/run/defaults.mk

vectors.o : vectors.hpp
wrapvectors.o : wrapvectors.h vectors.hpp
main.o : fruit.o wrapfvectors.o

# useful to isolate load error with just C++ code
maincpp : maincpp.o wrapvectors.o vectors.o
	$(CXX) $(CXXFLAGS) $^ -o $@ $(CLIBS)

testc.o : testc.c wrapvectors.h vectors.hpp

