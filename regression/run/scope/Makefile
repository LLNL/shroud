# Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and
# other Shroud Project Developers.
# See the top-level COPYRIGHT file for details.
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

include $(top)/regression/run/defaults.mk

INCLUDE = \
    -I$(top)/regression/reference/scope \
    -I$(top)/regression/run/scope \
    -I$(top)/regression/run/fruit

VPATH = \
    $(top)/regression/reference/scope \
    $(top)/regression/run/scope \
    $(top)/regression/run/fruit

OBJS = \
    scope.o \
    wrapfscope.o \
    wrapscope.o \
    fruit.o \
    main.o

scope : $(OBJS)
	$(FC) $(FFLAGS) $^ -o $@ $(FLIBS)

clean :
	rm -f $(OBJS) *.mod scope

scope.o : scope.hpp
wrapScope.o : wrapscope.h scope.hpp
main.o : fruit.o wrapfscope.o

# useful to isolate load error with just C++ code
maincpp : maincpp.o wrapscope.o
	$(CXX) $(CXXFLAGS) $(LIBS) $^ -o $@ $(CXXLIBS)

testc.o : testc.c wrapscope.h scope.hpp
testc : testc.o wrapscope.o scope.o
	$(CC) $(CFLAGS) $^ -o $@ $(CLIBS)
