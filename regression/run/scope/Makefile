# Copyright Shroud Project Developers. See LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
#
# #######################################################################
#
# run/scope
#
ifeq ($(top),)
top := ../../..
endif 

testdir = scope

INCLUDE = \
    -I$(top)/regression/reference/scope \
    -I$(top)/regression/run/scope

VPATH = \
    $(top)/regression/reference/scope \
    $(top)/regression/run/scope \
    $(top)/regression/run/fruit

C_OBJS = \
    scope.o \
    wrapscope.o
F_OBJS = \
    main.o \
    fruit.o \
    wrapfscope.o

include $(top)/regression/run/defaults.mk

scope.o : scope.hpp
wrapscope.o : wrapscope.h scope.hpp
main.o : fruit.o wrapfscope.o

# useful to isolate load error with just C++ code
maincpp : maincpp.o #wrapscope.o
	$(CXX) $(CXXFLAGS) $(LIBS) $^ -o $@ $(CXXLIBS)

testc.o : testc.c wrapscope.h scope.hpp
