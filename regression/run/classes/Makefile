# Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and
# other Shroud Project Developers.
# See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
# #######################################################################
#
# run/classes
#
ifeq ($(top),)
top := ../../..
endif 

include $(top)/regression/run/defaults.mk

INCLUDE = \
    -I$(top)/regression/reference/classes \
    -I$(top)/regression/run/classes

VPATH = \
    $(top)/regression/reference/classes \
    $(top)/regression/run/classes \
    $(top)/regression/run/fruit

C_OBJS = \
    classes.o \
    wrapclasses.o \
    wrapClass1.o \
    wrapClass2.o \
    wrapSingleton.o \
    wrapShape.o \
    wrapCircle.o \
    wrapData.o \
    utilclasses.o
F_OBJS = \
    wrapfclasses.o \
    fruit.o \
    main.o

classes : $(C_OBJS) $(F_OBJS)
	$(FC) $(FFLAGS) $^ -o $@ $(FLIBS)

clean :
	rm -f $(OBJS) *.mod classes

classes.o : classes.hpp
wrapClasses.o : wrapClasses.h classes.hpp
wrapClass1.o : wrapClass1.h classes.hpp
wrapClass2.o : wrapClass2.h classes.hpp
wrapSingleton.o : wrapSingleton.h classes.hpp
wrapData.o : wrapData.h wrapData.cpp
main.o : fruit.o wrapfclasses.o

# useful to isolate load error with just C++ code
maincpp : maincpp.o wrapclasses.o wrapClass1.o wrapSingleton.o classes.o
	$(CXX) $(CXXFLAGS) $(LIBS) $^ -o $@ $(CXXLIBS)

testc.o : testc.c wrapclasses.h wrapClass1.h classes.hpp
testc : testc.o $(C_OBJS)
	$(CC) $(CFLAGS) $^ -o $@ $(CLIBS)
