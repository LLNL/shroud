# Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and
# other Shroud Project Developers.
# See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
# #######################################################################
#
# run/types
#
include $(top)/regression/run/defaults.mk

INCLUDE = \
    -I$(top)/regression/reference/types \
    -I$(top)/regression/run/types \
    -I$(top)/regression/run/fruit

VPATH = \
    $(top)/regression/reference/types \
    $(top)/regression/run/types \
    $(top)/regression/run/fruit

C_OBJS = \
    types.o \
    wraptypes.o
F_OBJS = \
    wrapftypes.o \
    fruit.o \
    main.o

types : $(C_OBJS) $(F_OBJS)
	$(FC) $(FFLAGS) $^ -o $@ $(FLIBS)

clean :
	rm -f $(OBJS) *.mod types

types.o : types.hpp
wraptypes.o : wraptypes.h types.hpp
main.o : fruit.o wrapftypes.o types.o

# useful to isolate load error with just C++ code
maincpp : maincpp.o wraptypes.o types.o
	$(CXX) $(CXXFLAGS) $^ -o $@ $(CXXLIBS)

testc.o : testc.c wraptypes.h types.hpp
testc : testc.o $(C_OBJS)
	$(CC) $(CFLAGS) $^ -o $@ $(CLIBS)
