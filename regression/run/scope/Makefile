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
    wrapfcls1_enum.o \
    wrapfcls2_enum.o \
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
main.o : wrapfscope.o

# useful to isolate load error with just C++ code
maincpp : maincpp.o wrapscope.o scope.o
	$(CXX) $(CXXFLAGS) $(LIBS) $^ -o $@ $(CXXLIBS)

testc.o : testc.c wrapscope.h scope.hpp
testc : testc.o wrapscope.o scope.o
	$(CC) $(CFLAGS) $^ -o $@ $(CLIBS)

testscope.o : scope.hpp
testscope: testscope.o
	$(CXX) $^ -o $@
