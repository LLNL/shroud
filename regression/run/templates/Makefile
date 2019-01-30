# Copyright (c) 2018-2019, Lawrence Livermore National Security, LLC. 
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
    -I$(top)/regression/reference/templates \
    -I$(top)/regression/run/templates \
    -I$(top)/regression/run/fruit

VPATH = \
    $(top)/regression/reference/templates \
    $(top)/regression/run/templates \
    $(top)/regression/run/fruit

OBJS = \
    wraptemplates.o \
    wrapvectorforint.o \
    wrapvector_double.o \
    wrapfvector_int.o \
    wrapfvector_double.o \
    wrapftemplates.o \
    wrapfWorker.o \
    fruit.o \
    main.o

#    wrapuser.o
#    wrapfuser_int.o


templates : $(OBJS)
	$(FC) $(FFLAGS) $^ -o $@ $(FLIBS)

clean :
	rm -f $(OBJS) *.mod templates

#templates.o : templates.cpp templates.hpp
wrapvectorforint.o : wrapvectorforint.cpp wrapvector_int.h
wrapuser_int.o : wrapuser_int.cpp wrapuser_int.h typestemplates.h templates.hpp
wraptemplates.o : wraptemplates.cpp wraptemplates.h typestemplates.h templates.hpp
main.o : main.f wrapfvector_int.o wrapfuser_int.o wrapftemplates.o wrapfWorker.o
maincpp.o : maincpp.cpp templates.hpp

# useful to isolate load error with just C++ code
maincpp : maincpp.o
	$(CXX) $(CXXFLAGS) $^ -o $@ $(CXXLIBS)

testc.o : testc.c wrapvector_int.h
testc : testc.o wrapvectorforint.o
	$(CC) $(CFLAGS) $^ -o $@ $(CLIBS)
