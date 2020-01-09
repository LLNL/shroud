# Copyright (c) 2017-2020, Lawrence Livermore National Security, LLC and
# other Shroud Project Developers.
# See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
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
    wrapfpreprocess.o \
    wrappreprocess.o \
    wrapUser1.o \
    wrapUser2.o \
    fruit.o \
    main.o

preprocess : $(OBJS)
	$(FC) $(FFLAGS) $^ -o $@ $(FLIBS)

clean :
	rm -f $(OBJS) *.mod preprocess

wrappreprocess.o : preprocess.hpp
wrapUser1.o : wrapUser1.h preprocess.hpp wrapUser1.h
main.o : wrapfpreprocess.o wrappreprocess.o
