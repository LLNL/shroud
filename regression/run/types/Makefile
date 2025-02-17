# Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
# other Shroud Project Developers.
# See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
# #######################################################################
#
# run/types
#
testdir = types

INCLUDE = \
    -I$(top)/regression/reference/types \
    -I$(top)/regression/run/types

VPATH = \
    $(top)/regression/reference/types \
    $(top)/regression/run/types \
    $(top)/regression/run/fruit

C_OBJS = \
    types.o \
    wraptypes.o
F_OBJS = \
    main.o \
    fruit.o \
    wrapftypes.o

include $(top)/regression/run/defaults.mk

types.o : types.hpp
wraptypes.o : wraptypes.h types.hpp
main.o : fruit.o wrapftypes.o types.o

# useful to isolate load error with just C++ code
maincpp : maincpp.o wraptypes.o types.o
	$(CXX) $(CXXFLAGS) $^ -o $@ $(CXXLIBS)

testc.o : testc.c wraptypes.h

