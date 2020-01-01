# Copyright (c) 2017-2020, Lawrence Livermore National Security, LLC and
# other Shroud Project Developers.
# See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
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

C_OBJS = \
    tutorial.o \
    wrapTutorial.o \
    wrapClass1.o \
    wrapSingleton.o
F_OBJS = \
    wrapftutorial.o \
    fruit.o \
    main.o

tutorial : $(C_OBJS) $(F_OBJS)
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
testc : testc.o $(C_OBJS)
	$(CC) $(CFLAGS) $^ -o $@ $(CLIBS)
