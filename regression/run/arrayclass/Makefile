# Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
# other Shroud Project Developers.
# See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
#
# #######################################################################
#
# run/arrayclass
#
include $(top)/regression/run/defaults.mk

INCLUDE = \
    -I$(top)/regression/reference/arrayclass \
    -I$(top)/regression/run/arrayclass

VPATH = \
    $(top)/regression/reference/arrayclass \
    $(top)/regression/run/arrayclass \
    $(top)/regression/run/fruit

#    arrayclass.o   # This file is not created
OBJS = \
    wrapfarrayclass.o \
    wrapArrayWrapper.o \
    utilarrayclass.o \
    fruit.o \
    main.o

arrayclass : $(OBJS)
	$(FC) $(FFLAGS) $^ -o $@ $(CLIBS)

clean :
	rm -f $(OBJS) *.mod arrayclass

main.o : fruit.o wrapfarrayclass.o
