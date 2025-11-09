# Copyright Shroud Project Developers. See LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
# #######################################################################
#
# run/typemap64
#
ifeq ($(top),)
top := ../../..
endif 

testdir = typemap64

TEST_CFLAGS   = -DUSE_64BIT_INDEXTYPE -DUSE_64BIT_FLOAT
TEST_CXXFLAGS = -DUSE_64BIT_INDEXTYPE -DUSE_64BIT_FLOAT
TEST_FFLAGS   = -DUSE_64BIT_INDEXTYPE -DUSE_64BIT_FLOAT

INCLUDE = \
    -I$(top)/regression/reference/typemap \
    -I$(top)/regression/run/typemap

VPATH = \
    $(top)/regression/reference/typemap \
    $(top)/regression/run/typemap \
    $(top)/regression/run/fruit

C_OBJS = \
    typemap.o \
    utiltypemap.o \
    wraptypemap.o
F_OBJS = \
    main.o \
    fruit.o \
    wrapftypemap.o

include $(top)/regression/run/defaults.mk

typemap.o : typemap.hpp
wraptypemap.o : wraptypemap.h typemap.hpp
main.o : fruit.o wrapftypemap.o

# useful to isolate load error with just C++ code
maincpp : maincpp.o wraptypemap.o typemap.o
	$(CXX) $(CXXFLAGS) $(LIBS) $^ -o $@ $(CXXLIBS) -lm

testc.o : testc.c wraptypemap.h typemap.hpp
