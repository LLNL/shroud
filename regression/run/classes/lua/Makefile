# Copyright Shroud Project Developers. See LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
#
# #######################################################################
#
#  lua run/classes
#
testdir = classes

AS_SHARED := TRUE
include $(top)/regression/run/defaults.mk

INCLUDE = \
    -I$(top)/regression/reference/classes \
    -I$(top)/regression/run/classes \
    $(LUA_INC)

VPATH = \
    $(top)/regression/reference/classes \
    $(top)/regression/run/classes \
    $(top)/regression/run/classes/lua

OBJS = \
    classes.o \
    luaclassesmodule.o

all : classes.so # simple

classes.so : $(OBJS)
	$(CXX) $(LD_SHARED) -o $@ $^ $(LIBS)

# XXX testlua.c has some 5.2 features
simple : testlua.o $(OBJS)
	$(CC) -o $@ $^ $(LUA_LIB)
