# Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and
# other Shroud Project Developers.
# See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
#

include ../../defaults.mk
include $(rundir)/defaults.mk

INCLUDE = \
    -I$(rundir)/struct \
    -I$(rundir)/fruit

VPATH = \
    $(rundir)/struct \
    $(rundir)/fruit

OBJS = \
    struct.o \
    cstruct.o \
    struct_wrap.o \
    fruit.o \
    main.o

struct : $(OBJS)
	$(FC) $(FFLAGS) $^ -o $@ $(FLIBS)

cstruct.f90 : struct.i
	$(swig-fortran) -fortran struct.i

swig:
#	swig -fortran -o swigstruct_module.c struct.i
	$(swig-fortran) -fortran struct.i

test:
	./struct

clean :
	rm -f $(OBJS) *.mod struct

.PHONY : swig test clean
