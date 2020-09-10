# Copyright (c) 2017-2020, Lawrence Livermore National Security, LLC and
# other Shroud Project Developers.
# See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
#
# #######################################################################
#
#  lua run/classes
#
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
    luaClassesmodule.o

CXXFLAGS += $(SHARED)

all : classes.so # simple

classes.so : $(OBJS)
	$(CXX) $(LD_SHARED) -o $@ $^ $(LIBS)

# XXX testlua.c has some 5.2 features
simple : testlua.o $(OBJS)
	$(CC) -o $@ $^ $(LUA_LIB)

clean :
	rm -f *.so *.o simple
.PHONY : clean

print-debug:
	@echo LUA=$(LUA)
	@echo LUA_PREFIX=$(LUA_PREFIX)
	@echo LUA_INC=$(LUA_INC)
	@echo LUA_LIB=$(LUA_LIB)
