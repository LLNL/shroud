# Copyright (c) 2017-2020, Lawrence Livermore National Security, LLC and
# other Shroud Project Developers.
# See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
#
# #######################################################################
#
# run/references
#
include $(top)/regression/run/defaults.mk

INCLUDE = \
    -I$(top)/regression/reference/references \
    -I$(top)/regression/run/references

VPATH = \
    $(top)/regression/reference/references \
    $(top)/regression/run/references \
    $(top)/regression/run/fruit

#    references.o   # This file is not created
OBJS = \
    wrapfreferences.o \
    wrapreferences.o \
    wrapArrayWrapper.o \
    fruit.o \
    main.o

references : $(OBJS)
	$(FC) $(FFLAGS) $^ -o $@ $(CLIBS)

clean :
	rm -f $(OBJS) *.mod references

main.o : wrapfreferences.o
