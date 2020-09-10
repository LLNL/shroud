# Copyright (c) 2017-2020, Lawrence Livermore National Security, LLC and
# other Shroud Project Developers.
# See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
#
# #######################################################################
#
#  lua run/types
#
include $(top)/regression/run/defaults.mk

INCLUDE = \
    -I$(top)/regression/reference/types \
    -I$(top)/regression/run/types \
    $(LUA_INC)

VPATH = \
    $(top)/regression/reference/types \
    $(top)/regression/run/types \
    $(top)/regression/run/types/lua

OBJS = \
    types.o \
    luaTypesmodule.o

CXXFLAGS += $(SHARED)

all : types.so # simple

types.so : $(OBJS)
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
