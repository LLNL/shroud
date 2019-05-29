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
# run/preprocess
#
include $(top)/regression/run/defaults.mk

INCLUDE = \
    -I$(top)/regression/reference/preprocess \
    -I$(top)/regression/run/preprocess \
    -I$(top)/regression/run/fruit

VPATH = \
    $(top)/regression/reference/preprocess \
    $(top)/regression/run/preprocess \
    $(top)/regression/run/fruit

OBJS = \
    wrapfUser1.f \
    wrappreprocess.o \
    wrapUser1.o \
    fruit.o \
    main.o

preprocess : $(OBJS)
	$(FC) $(FFLAGS) $^ -o $@ $(FLIBS)

clean :
	rm -f $(OBJS) *.mod preprocess

preprocess.o : preprocess.hpp
wrappreprocess.o : preprocess.hpp
wrapUser1.o : wrapUser1.h preprocess.hpp
main.o : wrapfUser1.o
