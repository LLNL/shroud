# Copyright (c) 2017-2019, Lawrence Livermore National Security, LLC.
#
# Produced at the Lawrence Livermore National Laboratory
#
# LLNL-CODE-738041.
#
# All rights reserved.
#
# This file is part of Shroud.
#
# For details about use and distribution, please read LICENSE.
#
# #######################################################################
#
# run/tutorial
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

OBJS = \
    types.o \
    wrapftypes.o \
    wraptypes.o \
    fruit.o \
    main.o

types : $(OBJS)
	$(FC) $(FFLAGS) $^ -o $@ $(FLIBS)

clean :
	rm -f $(OBJS) *.mod types

types.o : types.hpp
wraptypes.o : wraptypes.h types.hpp
main.o : wrapftypes.o types.o

# useful to isolate load error with just C++ code
maincpp : maincpp.o wraptypes.o types.o
	$(CXX) $(CXXFLAGS) $^ -o $@ $(CXXLIBS)

testc.o : testc.c wraptypes.h types.hpp
testc : testc.o wraptypes.o types.o
	$(CC) $(CFLAGS) $^ -o $@ $(CLIBS)
