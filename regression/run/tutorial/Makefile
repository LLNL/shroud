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
ifeq ($(top),)
top := ../../..
endif 

include $(top)/regression/run/defaults.mk

INCLUDE = \
    -I$(top)/regression/reference/tutorial \
    -I$(top)/regression/run/tutorial \
    -I$(top)/regression/run/fruit

VPATH = \
    $(top)/regression/reference/tutorial \
    $(top)/regression/run/tutorial \
    $(top)/regression/run/fruit

OBJS = \
    tutorial.o \
    wrapftutorial.o \
    wrapTutorial.o \
    wrapClass1.o \
    wrapSingleton.o \
    fruit.o \
    main.o

tutorial : $(OBJS)
	$(FC) $(FFLAGS) $^ -o $@ $(FLIBS)

clean :
	rm -f $(OBJS) *.mod tutorial

tutorial.o : tutorial.hpp
wrapTutorial.o : wrapTutorial.h tutorial.hpp
wrapClass1.o : wrapClass1.h tutorial.hpp
wrapSingleton.o : wrapSingleton.h tutorial.hpp
main.o : wrapftutorial.o

# useful to isolate load error with just C++ code
maincpp : maincpp.o wrapTutorial.o wrapClass1.o wrapSingleton.o tutorial.o
	$(CXX) $(CXXFLAGS) $(LIBS) $^ -o $@ $(CXXLIBS)

testc.o : testc.c wrapTutorial.h wrapClass1.h tutorial.hpp
testc : testc.o wrapTutorial.o wrapClass1.o wrapSingleton.o tutorial.o
	$(CC) $(CFLAGS) $^ -o $@ $(CLIBS)
