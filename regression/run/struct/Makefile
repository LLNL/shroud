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
# run/struct
#
include $(top)/regression/run/defaults.mk

INCLUDE = \
    -I$(top)/regression/reference/struct \
    -I$(top)/regression/run/struct \
    -I$(top)/regression/run/fruit

VPATH = \
    $(top)/regression/reference/struct \
    $(top)/regression/run/struct \
    $(top)/regression/run/fruit

OBJS = \
    struct.o \
    wrapfstruct.o \
    wrapstruct.o \
    fruit.o \
    main.o

struct : $(OBJS)
	$(FC) $(FFLAGS) $^ -o $@ $(FLIBS)

clean :
	rm -f $(OBJS) *.mod struct

struct.o : struct.h
wrapstruct.o : wrapstruct.h struct.h
main.o : wrapfstruct.o
