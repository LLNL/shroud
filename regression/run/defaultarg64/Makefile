# Copyright Shroud Project Developers. See LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
# #######################################################################
#
# run/defaultarg64
#
ifeq ($(top),)
top := ../../..
endif 

testdir = defaultarg64

TEST_CFLAGS   = -DINDEXTYPE_SIZE=64
TEST_CXXFLAGS = -DINDEXTYPE_SIZE=64
TEST_FFLAGS   = -DINDEXTYPE_SIZE=64

INCLUDE = \
    -I$(top)/regression/reference/defaultarg \
    -I$(top)/regression/run/defaultarg

VPATH = \
    $(top)/regression/reference/defaultarg \
    $(top)/regression/run/defaultarg \
    $(top)/regression/run/fruit

C_OBJS = \
    defaultarg.o \
    wrapdefaultarg.o \
    wrapClass1.o \
    utildefaultarg.o
F_OBJS = \
    main.o \
    fruit.o \
    wrapfdefaultarg.o

include $(top)/regression/run/defaults.mk

defaultarg.o : defaultarg.cpp defaultarg.hpp
wrapdefaultarg.o : wrapdefaultarg.h defaultarg.hpp
wrapClass1.o : wrapClass1.cpp defaultarg.hpp wrapClass1.h
main.o : fruit.o wrapfdefaultarg.o

# useful to isolate load error with just C++ code
maincpp : maincpp.o defaultarg.o
	$(CXX) $(CXXFLAGS) $(LIBS) $^ -o $@ $(CXXLIBS)

testc.o : testc.c wrapdefaultarg.h defaultarg.hpp

