# Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
# other Shroud Project Developers.
# See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
# #######################################################################
#
# run/cdesc
#
ifeq ($(top),)
top := ../../..
endif 

include $(top)/regression/run/defaults.mk

INCLUDE = \
    -I$(top)/regression/reference/cdesc \
    -I$(top)/regression/run/cdesc

VPATH = \
    $(top)/regression/reference/cdesc \
    $(top)/regression/run/cdesc \
    $(top)/regression/run/fruit

C_OBJS = \
    cdesc.o \
    wrapcdesc.o
F_OBJS = \
    wrapfcdesc.o \
    utilcdesc.o \
    fruit.o \
    main.o

cdesc : $(C_OBJS) $(F_OBJS)
	$(FC) $(FFLAGS) $^ -o $@ $(FLIBS)

clean :
	rm -f $(OBJS) *.mod cdesc

cdesc.o : cdesc.hpp
wrapcdesc.o : wrapcdesc.h cdesc.hpp
main.o : fruit.o wrapfcdesc.o

# useful to isolate load error with just C++ code
maincpp : maincpp.o wrapcdesc.o cdesc.o
	$(CXX) $(CXXFLAGS) $(LIBS) $^ -o $@ $(CXXLIBS)

testc.o : testc.c wrapcdesc.h cdesc.hpp
testc : testc.o $(C_OBJS)
	$(CC) $(CFLAGS) $^ -o $@ $(CLIBS)
