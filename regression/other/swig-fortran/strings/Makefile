# Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and
# other Shroud Project Developers.
# See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
#

include ../../defaults.mk
include $(rundir)/defaults.mk

INCLUDE = \
    -I$(rundir) \
    -I$(rundir)/strings \
    -I$(rundir)/fruit

VPATH = \
    $(rundir)/strings \
    $(rundir)/fruit

OBJS = \
    strings.o \
    strings_mod.o \
    strings_wrap.o \
    fruit.o \
    main.o

strings : $(OBJS)
	$(FC) $(FFLAGS) $^ -o $@ $(FLIBS)

swig:
#	swig -fortran -o swigstrings_module.c strings.i
	$(swig-fortran)  -fortran -c++ strings.i

test:
	./strings

clean :
	rm -f $(OBJS) *.mod strings *~

.PHONY : swig test clean
