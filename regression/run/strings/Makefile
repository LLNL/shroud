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
# run/strings
#
include $(top)/regression/run/defaults.mk

INCLUDE = \
    -I$(top)/regression/reference/strings \
    -I$(top)/regression/run/strings \
    -I$(top)/regression/run/fruit

VPATH = \
    $(top)/regression/reference/strings \
    $(top)/regression/run/strings \
    $(top)/regression/run/fruit

OBJS = \
    strings.o \
    wrapfstrings.o \
    wrapstrings.o \
    fruit.o \
    main.o

strings : $(OBJS)
	$(FC) $(FFLAGS) $^ -o $@ $(FLIBS)

clean :
	rm -f $(OBJS) *.mod strings

main.o : wrapfstrings.o
