# Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
# other Shroud Project Developers.
# See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
#
# #######################################################################
#
#  Compile the lua tutorial
#
include $(top)/regression/run/defaults.mk

INCLUDE = \
    -I$(top)/regression/reference/tutorial \
    -I$(top)/regression/run/tutorial \
    $(LUA_INC)

VPATH = \
    $(top)/regression/reference/tutorial \
    $(top)/regression/run/tutorial \
    $(top)/regression/run/tutorial/lua

OBJS = \
    tutorial.o \
    luaTutorialmodule.o

CXXFLAGS += $(SHARED)

all : tutorial.so # simple

tutorial.so : $(OBJS)
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
