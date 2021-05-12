# Copyright (c) 2017-2021, Lawrence Livermore National Security, LLC and
# other Shroud Project Developers.
# See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
#

include ../../defaults.mk
include $(rundir)/defaults.mk

INCLUDE = \
    -I$(rundir)/pointers \
    -I$(rundir)/fruit

VPATH = \
    $(rundir)/pointers \
    $(rundir)/fruit

OBJS = \
    pointers.o \
    pointers_mod.o \
    pointers_wrap.o \
    fruit.o \
    main.o

pointers : $(OBJS)
	$(FC) $(FFLAGS) $^ -o $@ $(FLIBS)

swig:
#	swig -fortran -o swigpointers_module.c pointers.i
	$(swig-fortran)  -fortran pointers.i

test:
	./pointers

clean :
	rm -f $(OBJS) *.mod pointers *~

.PHONY : swig test clean
