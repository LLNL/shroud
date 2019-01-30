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
# run/clibrary
#
include $(top)/regression/run/defaults.mk

INCLUDE = \
    -I$(top)/regression/reference/clibrary \
    -I$(top)/regression/run/clibrary \
    -I$(top)/regression/run/fruit

VPATH = \
    $(top)/regression/reference/clibrary \
    $(top)/regression/run/clibrary \
    $(top)/regression/run/fruit

OBJS = \
    clibrary.o \
    wrapfclibrary.o \
    wrapClibrary.o \
    fruit.o \
    main.o

clibrary : $(OBJS)
	$(FC) $(FFLAGS) $^ -o $@ $(CLIBS)

clean :
	rm -f $(OBJS) *.mod clibrary

main.o : wrapfclibrary.o
