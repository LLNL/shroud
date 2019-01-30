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
# run/pointers
#
include $(top)/regression/run/defaults.mk

INCLUDE = \
    -I$(top)/regression/reference/pointers \
    -I$(top)/regression/run/pointers \
    -I$(top)/regression/run/fruit

VPATH = \
    $(top)/regression/reference/pointers \
    $(top)/regression/run/pointers \
    $(top)/regression/run/fruit

OBJS = \
    pointers.o \
    wrapfpointers.o \
    wrappointers.o \
    fruit.o \
    main.o

pointers : $(OBJS)
	$(FC) $(FFLAGS) $^ -o $@ $(CLIBS)

clean :
	rm -f $(OBJS) *.mod pointers

main.o : wrapfpointers.o
