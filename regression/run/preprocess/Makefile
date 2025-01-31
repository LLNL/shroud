# Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
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
    -I$(top)/regression/run/preprocess

VPATH = \
    $(top)/regression/reference/preprocess \
    $(top)/regression/run/preprocess \
    $(top)/regression/run/fruit

OBJS = \
    wrapfpreprocess.o \
    wrapUser1.o \
    wrapUser2.o \
    utilpreprocess.o \
    fruit.o \
    main.o

preprocess : $(OBJS)
	$(FC) $(FFLAGS) $^ -o $@ $(FLIBS)

clean :
	rm -f $(OBJS) *.mod preprocess

wrappreprocess.o : preprocess.hpp
wrapUser1.o : wrapUser1.h preprocess.hpp wrapUser1.h
main.o : fruit.o wrapfpreprocess.o wrappreprocess.o
