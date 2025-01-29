# Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and
# other Shroud Project Developers.
# See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
#
# run/ccomplex
#
include $(top)/regression/run/defaults.mk

INCLUDE = \
    -I$(top)/regression/reference/ccomplex \
    -I$(top)/regression/run/ccomplex

VPATH = \
    $(top)/regression/reference/ccomplex \
    $(top)/regression/run/ccomplex \
    $(top)/regression/run/fruit

OBJS = \
    ccomplex.o \
    wrapfccomplex.o \
    utilccomplex.o \
    fruit.o \
    main.o

ccomplex : $(OBJS)
	$(FC) $(FFLAGS) $^ -o $@ $(CLIBS)

clean :
	rm -f $(OBJS) *.mod ccomplex

main.o : fruit.o wrapfccomplex.o
