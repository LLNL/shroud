# Copyright Shroud Project Developers. See LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
#

testdir = struct
tester : $(testdir)
.PHONY : tester

INCLUDE = \
    -I$(rundir)/struct \
    -I$(rundir)/fruit

VPATH = \
    $(rundir)/struct \
    $(rundir)/fruit

F_OBJS = \
    cstruct.o \
    fruit.o \
    main.o

C_OBJS = \
    struct.o \
    struct_wrap.o

include ../../defaults.mk
include $(rundir)/defaults.mk

cstruct.f90 : struct.i
	$(swig-fortran) -fortran struct.i

swig:
#	swig -fortran -o swigstruct_module.c struct.i
	$(swig-fortran) -fortran struct.i

test:
	./struct

.PHONY : swig test
