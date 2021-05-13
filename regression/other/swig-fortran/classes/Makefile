# Copyright (c) 2017-2021, Lawrence Livermore National Security, LLC and
# other Shroud Project Developers.
# See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
#

include ../../defaults.mk
include $(rundir)/defaults.mk

INCLUDE = \
    -I$(rundir)/classes \
    -I$(rundir)/fruit

VPATH = \
    $(rundir)/classes \
    $(rundir)/fruit

OBJS = \
    classes.o \
    classes_mod.o \
    classes_wrap.o \
    fruit.o \
    main.o

classes : $(OBJS)
	$(FC) $(FFLAGS) $^ -o $@ $(FLIBS) $(CXXLIBS)

swig:
#	swig -fortran -o swigclasses_module.c classes.i
	$(swig-fortran)  -fortran -c++ classes.i

test:
	./classes

clean :
	rm -f $(OBJS) *.mod classes *~

.PHONY : swig test clean
